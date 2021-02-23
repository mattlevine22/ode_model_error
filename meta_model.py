import os
import numpy as np
import pickle
from scipy import signal # for periodogram
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2
from scipy.linalg import block_diag
from scipy.integrate import trapz, quad_vec
from scipy.interpolate import CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

from statsmodels.tsa.stattools import acf

# from scipy.stats import loguniform
import time
from functools import partial
print = partial(print, flush=True)
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from odelibrary import my_solve_ivp

# matt utils
from utils import file_to_dict, optimizer_as_df
from plotting_utils import *
from computation_utils import *

#BayesianOptimization
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


import pandas as pd

#memory tracking
# import psutil


import pdb

class IDK(object):
	def __init__(self, settings, default_esn_settings='./Config/esn_default_params.json', solver_dict='./Config/solver_settings.json'):
		# load default settings
		params = file_to_dict(default_esn_settings)
		# add extra settings and replace defaults as needed
		params.update(settings)

		# load solver settings and add them to params
		foo = file_to_dict(solver_dict)
		params['solver_dict'] = foo

		for key in params:
			exec('self.{} = params["{}"]'.format(key, key))

		#
		if self.useNTrain:
			self.tTrain = self.NTrain * self.dt


		# specify interpolation method for x(t), which is needed if solving ZY
		# integrals using a quadrature that samples off of the data sample times t_k
		if self.diff in ['Spline', 'TrueDeriv']:
			self.interp = 'Spline'
		else:
			self.interp = 'Linear'

		# for now, ignore the fixed test dt and test at training data dt
		self.dt_test = self.dt

		self.scaler = scaler(tt=self.scaler_tt, tt_derivative=self.scaler_tt_derivatives, component_wise=self.component_wise)
		self.fig_dir = os.path.join(self.saving_path, params["fig_dir"])
		self.model_dir = os.path.join(self.saving_path, params["model_dir"])
		self.logfile_dir = os.path.join(self.saving_path, params["logfile_dir"])

		# randomly initialize at trainNumber
		np.random.seed(self.trainNumber)

		self.f0only = int(self.modelType=='f0only')
		self.dynamics_length = 1

		### quadrature rules
		self.default_quad_limit = self.tTrain / self.quad_limit_factor

		#### setup solver and quad limit
		self.set_fidelity("default")

		####### Add physical mechanistic rhs "f0" ##########
		if self.stateType=='state':
			self.rf_error_input = 0
		elif self.stateType=='stateAndPred':
			self.rf_error_input = 1
		else:
			raise ValueError('stateType unrecognized')

		self.doResidual = params["doResidual"]
		self.usef0 = params["usef0"]
		if self.usef0:
			self.f0 = params["f0"]
		else:
			self.f0 = 0

		if self.diff=='TrueDeriv':
			self.fTRUE = params["fTRUE"]

		# count the augmented input dimensions (i.e. [x(t); f0(x(t))-x(t)])
		if self.component_wise:
			self.input_dim_rf = 1 + self.rf_error_input
			self.output_dim_rf = 1
		else:
			self.input_dim_rf = (1 + self.rf_error_input)*self.input_dim
			self.output_dim_rf = self.input_dim

		os.makedirs(self.model_dir, exist_ok=True)
		os.makedirs(self.fig_dir, exist_ok=True)
		os.makedirs(self.logfile_dir, exist_ok=True)
		print('FIGURE PATH:', self.fig_dir)

	def set_fidelity(self, style):
		print('Fidelity:', style)
		try:
			self.solver_settings = self.solver_dict[style]
			if style=='Euler':
				self.solver_settings['dt'] = self.dt * self.solver_settings['dt_frac']
			elif style=='lowfi':
				self.quad_limit = self.default_quad_limit / 10
			else:
				self.quad_limit = self.default_quad_limit
		except:
			raise ValueError('fidelity style not recognized')

	def train(self):
		# if self.dont_redo and os.path.exists(self.saving_path + self.model_dir + "/data.pickle"):
		# 	raise ValueError('Model has already run for this configuration. Exiting with an error.')

		self.start_time = time.time()

		## DATA
		with open(self.train_data_path, "rb") as file:
			# Pickle the "data" dictionary using the highest protocol available.
			data = pickle.load(file)
			self.dt_rawdata = data["dt"]
			self.train_input_sequence = self.scaler.scaleData(self.subsample(x=data["u_train"][self.trainNumber, :, :self.input_dim], t_end=self.tTrain))
			self.train_input_sequence_dot = self.scaler.scaleXdot(self.subsample(x=data["udot_train"][self.trainNumber, :, :self.input_dim], t_end=self.tTrain))
			del data

		if self.f0only:
			self.saveModel()
			return

		self.setup_timeseries()

		# Do a hyperparameter optimization using a validation step
		self.set_fidelity('hifi')
		if self.validate_rf:
			# switch to lowfi quadrature for cheap validation runs

			# first learn W, b using default regularization
			pbounds = {'rf_Win_bound': (0,10),
						'rf_bias_bound': (0,20)}
			optimizer = BayesianOptimization(f=self.validation_function,
											pbounds=pbounds,
											random_state=1)
			log_path = os.path.join(self.logfile_dir, "BayesOpt_log.json")
			logger = JSONLogger(path=log_path) #conda version doesnt have RESET feature
			optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
			# for log_reg_rf in log_reg_list:
			# 	optimizer.probe(params={"log_regularization_RF": log_reg_rf}, lazy=True)
			optimizer.maximize(init_points=5, n_iter=30, acq='ucb')
			best_param_dict = optimizer.max['params']
			best_quality = optimizer.max['target']
			print("Optimal parameters:", best_param_dict, '(quality = {})'.format(best_quality))
			# re-setup things with optimal parameters (new realization using preferred hyperparams)
			self.set_BO_keyval(best_param_dict)

		# create Y, Z using chosen RF-parameters
		self.setup_the_learning()

		# select optimal regularization under the chosen Y, Z
		if self.validate_regularization:
			# now fix the optimal W,b and the learned Y,Z... just learn the optimal regularization for the inversion
			lambda_validation_f = lambda **kwargs: self.validation_function(setup_learning=False, **kwargs)
			pbounds = {'log_regularization_RF': (-20, 0)}
			optimizer = BayesianOptimization(f=lambda_validation_f,
											pbounds=pbounds,
											random_state=1)
			log_path = os.path.join(self.logfile_dir, "BayesOpt_reg_log.json")
			logger = JSONLogger(path=log_path) #conda version doesnt have RESET feature
			optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
			optimizer.probe(params={"log_regularization_RF": np.log10(self.regularization_RF)}, lazy=True)
			optimizer.maximize(init_points=5, n_iter=30, acq='ucb')
			best_param_dict = optimizer.max['params']
			best_quality = optimizer.max['target']
			print("Optimal parameters:", best_param_dict, '(quality = {})'.format(best_quality))
			self.set_BO_keyval(best_param_dict)

			# plot results from validation runs
			df = optimizer_as_df(optimizer)
			self.makeValidationPlots(df=df, plot_nm='reg')

		# solve for the final Y,Z, regI and save
		self.doNewSolving()
		self.saveModel()

	def setup_timeseries(self):
		self.xdot_vec_TRUE = np.copy(self.train_input_sequence_dot)
		self.x_vec = np.copy(self.train_input_sequence)
		self.x_vec_raw = self.scaler.descaleData(self.x_vec)
		t_vec = self.dt*np.arange(self.x_vec.shape[0])

		if self.interp=='Spline':
			self.x_spline = [CubicSpline(x=t_vec, y=self.x_vec[:,k]) for k in range(self.input_dim)]
		elif self.interp=='Linear':
			pass
		else:
			raise ValueError('Interpolation method not recognized.')

		# get derivative
		if self.diff == 'TrueDeriv': # use true derivative
			self.xdot_vec = np.copy(self.xdot_vec_TRUE)
			if self.derivNoiseSD:
				self.xdot_vec += np.random.normal(scale=self.derivNoiseSD, size=self.xdot_vec.shape)
		elif self.diff == 'Spline': # do spline derivative
			self.xdot_spline = [CubicSpline(x=t_vec, y=self.x_vec[:,k]).derivative() for k in range(self.input_dim)]
			# get m(t) for all time JUST to have its statistics for normalization ahead of time
			self.xdot_vec = np.array([self.xdot_spline[k](t_vec) for k in range(self.input_dim)]).T
		elif self.diff == 'Euler': # do euler derivative
			self.xdot_vec = np.zeros(self.xdot_vec_TRUE.shape)
			for k in range(len(t_vec)-1):
				self.xdot_vec[k] = (self.x_vec[k+1] - self.x_vec[k]) / self.dt
			# need to deal with boundary issue
			self.xdot_vec[-1] = self.xdot_vec[-2]
		elif self.diff == 'FD1forward':
			self.xdot_vec = np.zeros(self.xdot_vec_TRUE.shape)
			for k in range(len(t_vec)-1):
				self.xdot_vec[k] = (self.x_vec[k+1] - self.x_vec[k]) / self.dt
			# do backward euler for last element
			self.xdot_vec[-1] = (self.x_vec[-1] - self.x_vec[-2]) / self.dt
		elif self.diff == 'FD1backward':
			self.xdot_vec = np.zeros(self.xdot_vec_TRUE.shape)
			for k in range(1,len(t_vec)):
				self.xdot_vec[k] = (self.x_vec[k] - self.x_vec[k-1]) / self.dt
			# do forward euler for first element
			self.xdot_vec[0] = (self.x_vec[1] - self.x_vec[0]) / self.dt
		elif self.diff == 'FD2central':
			self.xdot_vec = np.zeros(self.xdot_vec_TRUE.shape)
			for k in range(1, len(t_vec)-1):
				self.xdot_vec[k] = (self.x_vec[k+1] - self.x_vec[k-1]) / (2*self.dt)
			# do forward euler for first element
			self.xdot_vec[0] = (self.x_vec[1] - self.x_vec[0]) / self.dt
			# do backward euler for last element
			self.xdot_vec[-1] = (self.x_vec[-1] - self.x_vec[-2]) / self.dt
		elif self.diff == 'FD4central':
			self.xdot_vec = np.zeros(self.xdot_vec_TRUE.shape)
			for k in range(2, len(t_vec)-2):
				self.xdot_vec[k] = (-self.x_vec[k+2] + 8*self.x_vec[k+1] - 8*self.x_vec[k-1] + self.x_vec[k-2]) / (12*self.dt)
			# do forward euler for first element
			self.xdot_vec[0] = (self.x_vec[1] - self.x_vec[0]) / self.dt
			# do central diff for second element
			self.xdot_vec[1] = (self.x_vec[2] - self.x_vec[0]) / (2*self.dt)
			# do central diff for second-to-last element
			self.xdot_vec[-2] = (self.x_vec[-1] - self.x_vec[-3]) / (2*self.dt)
			# do backward euler for last element
			self.xdot_vec[-1] = (self.x_vec[-1] - self.x_vec[-2]) / self.dt
		else:
			print('Not differentiating.')
			self.differentiation_error = None
			return

		self.differentiation_error = np.mean((self.xdot_vec-self.xdot_vec_TRUE)**2)
		print('|XdotInferred - XdotTRUE|=', self.differentiation_error)


	def setup_the_learning(self):

		if ('rhs' in self.modelType) and ('interp' in self.costIntegration):
			self.set_random_weights()
			self.continuousInterpRF()
		elif ('datagrid' in self.costIntegration) or ('Psi' in self.modelType):
			rf_input, x_output, x_input_descaled = self.get_regression_IO()
			if 'GP' in self.modelType:
				if self.component_wise:
					raise ValueError('component wise not yet set up for Euler')
				GP_ker = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e+6)) + WhiteKernel(1.0, (1e-10, 1e6))
				my_gpr = GaussianProcessRegressor(
					kernel = GP_ker,
					n_restarts_optimizer = 15,
					alpha = 1e-10)
				self.gpr = my_gpr.fit(X=rf_input, y=x_output)
			else:
				self.set_random_weights()
				if self.component_wise:
					Y = []
					Z = []
					for k in range(self.input_dim):
						# allocate, reshape, normalize, and save solutions
						Q = np.array([self.q_t(rf_input[k,i]) for i in range(rf_input.shape[1])])
						Z.append(Q.T @ Q)
						Y.append((Q.T @ x_output[k])[:,None])
						self.reg_dim = Q.shape[1]
					self.Y = np.vstack(Y)
					self.Z = np.vstack(Z)
				else:
					# allocate, reshape, normalize, and save solutions
					Q = np.array([self.q_t(rf_input[i]) for i in range(rf_input.shape[0])])
					self.Z = Q.T @ Q
					self.Y = Q.T @ x_output
					self.reg_dim = self.Z.shape[0]

				self.Z /= x_input_descaled.shape[0]
				self.Y /= x_input_descaled.shape[0]
				print('|Z| =', np.mean(self.Z**2))
				print('|Y| =', np.mean(self.Y**2))

	def test(self):
		for fidelity in ['hifi', 'Euler', 'hifiPlus']:
			self.set_fidelity(fidelity)

			# evaluate trajectory performance
			test_eval = self.testingOnSet(setnm='test', fidelity_name=fidelity, do_inv=True)

			self.write_stats(pd_stat=test_eval, stat_name='test_eval_{}'.format(fidelity))
			if fidelity=='hifi':
				self.write_stats(pd_stat=test_eval, stat_name='test_eval')
			print('Mean:\n', test_eval.mean())
			# print('SD:', test_eval.std())

	def validate(self):
		# self.testingOnTrainingSet()
		validate_eval = self.testingOnSet(setnm='validate', do_plots=False)
		print('Mean:\n', validate_eval.mean())
		# print('SD:', validate_eval.std())
		return validate_eval

	def set_BO_keyval(self, my_dict):
		for key in my_dict:
			if "log_" in key:
				my_val = 10**(float(my_dict[key]))
				my_varnm = key.strip('log_')
			else:
				my_val = my_dict[key]
				my_varnm = key
			exec("self.{varnm} = {val}".format(varnm=my_varnm, val=my_val))

	def validation_function(self, setup_learning=True, **kwargs):
		# self.regularization_RF = 10**(float(kwargs["log_regularization_RF"]))
		self.set_BO_keyval(kwargs)
		if setup_learning:
			self.setup_the_learning()
		self.doNewSolving(do_plots=False)
		quality_df = self.validate()
		quality = quality_df.t_valid_005.mean() + 0.0001*np.random.randn()
		return quality


	def testingOnSet(self, setnm, do_plots=True, fidelity_name='', do_inv=False):
		with open(self.test_data_path, "rb") as file:
			data = pickle.load(file)
			self.dt_rawdata = data["dt"]
			eval_data = data["u_{}".format(setnm)][:, :, :self.input_dim]
			del data

		n_traj = eval_data.shape[0]
		test_eval = []
		# loop over test sets
		print('Evaluating', n_traj, setnm, 'sets')
		for n in range(n_traj):
			test_input_sequence = self.subsample(x=eval_data[n], t_end=self.t_test, dt_subsample=self.dt_test)
			eval_dict = self.eval(input_sequence=test_input_sequence, t_end=self.t_test, set_name=setnm+fidelity_name+str(n), traj_plots=do_plots, inv_plots=False)
			test_eval.append(eval_dict)

		# regroup test_eval
		test_eval = pd.DataFrame(test_eval)

		if do_inv:
			setnm = "inv_meas"
			with open(self.test_data_path, "rb") as file:
				data = pickle.load(file)
				self.dt_rawdata = data["dt"]
				inv_data = data["u_{}".format(setnm)][0, :, :self.input_dim]
				del data

			print('Evaluating', setnm, 'set')
			inv_input_sequence = self.subsample(x=inv_data, t_end=self.t_inv, dt_subsample=self.dt_test)
			inv_dict = self.eval(input_sequence=inv_input_sequence, t_end=self.t_inv, set_name=setnm+fidelity_name, traj_plots=False, inv_plots=do_plots, inv_stats=True)

		for key in ["kl_all", "kl_mean", "acf_error"]:
			test_eval[key] = inv_dict[key]

		return test_eval
		# rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred, hidden_all = self.eval(test_input_sequence, dt, "TEST")

	def eval(self, input_sequence, t_end, set_name, traj_plots=True, inv_plots=False, inv_stats=False):
		# allocate ic and target
		ic = self.scaler.scaleData(input_sequence[0], reuse=1)
		target = input_sequence

		# get predictions
		prediction = self.make_predictions(ic=ic, t_end=t_end)
		prediction = self.scaler.descaleData(prediction)
		eval_dict = computeErrors(target, prediction, self.scaler.data_std, dt=self.dt_test)

		# add hyperparameters to test_eval.pickle output
		try:
			for my_varnm in ["regularization_RF", "rf_Win_bound", "rf_bias_bound", "differentiation_error"]:
				exec("val = self.{varnm}".format(varnm=my_varnm))
				eval_dict[my_varnm] = val
		except:
			pass

		if traj_plots:
			self.makeTrajPlots(true_traj=target, predicted_traj=prediction, set_name=set_name)

		if inv_stats:
			self.saveInvStats(true_traj=target, predicted_traj=prediction, set_name=set_name, eval_dict=eval_dict)

		return eval_dict

	def make_predictions(self, ic, t_end):
		if 'Psi' in self.modelType:
			prediction = []
			prediction.append(ic)
			n_steps = int(t_end / self.dt_test)
			n_discrete_iters = int(self.dt_test / self.dt)
			for n in range(n_steps):
				for m in range(n_discrete_iters):
					ic = self.predict_next(x_input=ic)
				prediction.append(ic)
			prediction = np.array(prediction)
		elif 'rhs' in self.modelType or self.f0only:
			N = int(t_end / self.dt_test) + 1
			t_eval = self.dt_test*np.arange(N)
			t_span = [t_eval[0], t_eval[-1]]
			prediction = my_solve_ivp(ic=ic, f_rhs=self.rhs, t_eval=t_eval, t_span=t_span, settings=self.solver_settings)
		return prediction


	def psi0(self, ic, t0=0):
		t_span = [t0, t0+self.dt]
		t_eval = np.array([t0+self.dt])
		pred = my_solve_ivp(ic=ic, f_rhs=self.f0, t_eval=t_eval, t_span=t_span, settings=self.solver_settings)
		return pred

	def predict_next(self, x_input, t0=0):
		u_next = np.zeros(self.input_dim)


		if 'Psi' in self.modelType:
			if self.doResidual or self.usef0:
				pred = self.scaler.scaleData(self.psi0(self.scaler.descaleData(x_input)), reuse=1)
			else:
				pred = np.zeros(self.input_dim)

			if self.component_wise:
				for k in range(self.input_dim):
					if self.rf_error_input:
						rf_input = np.hstack((x_input[k,None], pred[k,None]))
					else:
						rf_input = x_input[k,None]
					if 'GP' in self.modelType:
						u_next[k] = pred[k] + self.gpr.predict(rf_input)
					else:
						u_next[k] = pred[k] + self.W_out_markov @ self.q_t(rf_input)
			else:
				if self.rf_error_input:
					rf_input = np.hstack((x_input, pred))
				else:
					rf_input = x_input
				if 'GP' in self.modelType:
					u_next = pred + self.gpr.predict(rf_input)
				else:
					u_next = pred + self.W_out_markov @ self.q_t(rf_input)
		elif 'rhs' in self.modelType:
			t_span = [t0, t0+self.dt]
			t_eval = np.array([t0+self.dt])
			u_next = my_solve_ivp(ic=x_input, f_rhs=self.rhs, t_span=t_span, t_eval=t_eval, settings=self.solver_settings)
		elif self.modelType=='f0only':
			u_next = self.scaler.scaleData(self.psi0(self.scaler.descaleData(x_input)), reuse=1)
		else:
			raise ValueError('modelType not recognized')

		return np.squeeze(u_next)

	def rhs(self, t0, u0):
		#u0 is in normalized coordinates but dx is in UNnormalized coordinates
		x_input = u0[:self.input_dim]

		# pdb.set_trace()
		# rf_input_NEW, _, x_input_descaled_NEW = self.get_regIO(x_input=x_input)

		# add mechanistic rhs
		if self.usef0:
			f0 = self.scaler.scaleXdot(self.f0(t0, self.scaler.descaleData(x_input)))
			if self.modelType=='f0only':
				return f0

		f_error_markov = np.zeros(self.input_dim)
		if self.component_wise:
			for k in range(self.input_dim):
				if self.rf_error_input:
					rf_input = np.hstack((x_input[k,None], f0[k,None]))
				else:
					rf_input = x_input[k,None]
				f_error_markov[k] = self.W_out_markov @ self.q_t(rf_input)
		else:
			if self.rf_error_input:
				rf_input = np.hstack((x_input, f0))
			else:
				rf_input = x_input
			f_error_markov = self.W_out_markov @ self.q_t(rf_input)

		# total rhs for x
		if self.doResidual:
			dx = f0 + self.scaler.descaleM(f_error_markov)
		else:
			dx = self.scaler.descaleM(f_error_markov)

		return dx

	def makeValidationPlots(self, df, plot_nm=''):
		x_names = df.columns[df.columns!='target']
		for i in range(len(x_names)):
			xnm = x_names[i]
			fig_path = os.path.join(self.fig_dir, "validation_{}_{}.png".format(plot_nm, xnm))
			fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
			ax.scatter(df[xnm], df.target, color='blue')
			ax.set_xlabel(xnm)
			ax.set_ylabel('target')
			plt.suptitle('Validation Plots')
			plt.savefig(fig_path)
			plt.close()


	def makeTrajPlots(self, true_traj, predicted_traj, set_name=''):
		n_times = true_traj.shape[0] # self.X
		time_vec = np.arange(n_times)*self.dt
		mse = np.mean( (true_traj - predicted_traj)**2)

		# plot dynamics over time
		fig_path = os.path.join(self.fig_dir, "timewise_fits_{}.png".format(set_name))
		fig, ax = plt.subplots(nrows=self.input_dim, ncols=1,figsize=(12, 12))
		for k in range(self.input_dim):
			ax[k].set_ylabel(r"$Y_{k}$".format(k=k), fontsize=12)
			ax[k].plot(time_vec, true_traj[:,k], color='black',label='true trajectory')
			ax[k].plot(time_vec, predicted_traj[:,k], color='blue', label='predicted trajectory')
		ax[-1].legend()
		plt.suptitle('Timewise fits with total MSE {mse:.5}'.format(mse=mse))
		plt.savefig(fig_path)
		plt.close()

		# plot errors over time
		fig_path = os.path.join(self.fig_dir, "timewise_errors_{}.png".format(set_name))
		fig, ax = plt.subplots(nrows=self.input_dim, ncols=1,figsize=(12, 12))
		for k in range(self.input_dim):
			ax[k].set_ylabel(r"$Y_{k}$".format(k=k), fontsize=12)
			ax[k].plot(time_vec, true_traj[:,k] - predicted_traj[:,k], linewidth=2)
		plt.suptitle('Timewise errors with total MSE {mse:.5}'.format(mse=mse))
		plt.savefig(fig_path)
		plt.close()

	def saveInvStats(self, true_traj, predicted_traj, eval_dict, set_name=''):
		T_acf = 10
		nlags = int(T_acf/self.dt) - 1
		acfgrid = np.arange(0, T_acf, self.dt)

		if self.f0_name=='L63':
			ncols = self.input_dim
			data = []
			for k in range(self.input_dim):
				# INVARIANT MEASURE
				xgrid, Papprox, Ptrue = kdegrid(Xtrue=true_traj[:,k], Xapprox=predicted_traj[:,k], kde_func=kde_scipy, gridsize=1000)

				# AUTOCORRELATION FUNCTION
				acfapprox = acf(predicted_traj[:,k], fft=True, nlags=nlags) #look at first component
				acftrue = acf(true_traj[:,k], fft=True, nlags=nlags) #look at first component
				acferr = np.mean((acftrue - acfapprox)**2)

				data.append({"xgrid": xgrid,
							"P_approx": Papprox,
							"P_true": Ptrue,
							"acf_grid": acfgrid,
							"acf_approx": acfapprox,
							"acf_true": acftrue,
							"acf_err": acferr})


				fig_path = os.path.join(self.fig_dir, "kde_{}.png".format(set_name))
				fig, ax = plt.subplots(nrows=1, ncols=ncols,figsize=(24, 12))
				for k in range(ncols):
					sns.kdeplot(true_traj[:,k], ax=ax[k], label='True', linewidth=2)
					sns.kdeplot(predicted_traj[:,k], ax=ax[k], label='Predicted', linewidth=2)
					ax[k].set_ylabel("Probability", fontsize=12)
				ax[-1].legend()
				plt.suptitle('KDE with KL-Divergence={kl:.5}'.format(kl=eval_dict['kl_all']))
				plt.savefig(fig_path)
				plt.close()

				fig_path = os.path.join(self.fig_dir, "acf_{}.png".format(set_name))
				fig, ax = plt.subplots(nrows=ncols, ncols=1,figsize=(14, 8))
				for k in range(ncols):
					ax[k].plot(acfgrid, acftrue, label='True', linewidth=2)
					ax[k].plot(acfgrid, acfapprox, label='Predicted', linewidth=2)
					ax[k].set_ylabel("ACF", fontsize=12)
				ax[-1].legend()
				plt.suptitle('ACF with error={acf:.5}'.format(acf=acferr))
				plt.savefig(fig_path)
				plt.close()

		elif self.f0_name=='L96M':
			# INVARIANT MEASURE
			xgrid, Papprox, Ptrue = kdegrid(Xtrue=true_traj.reshape(-1), Xapprox=predicted_traj.reshape(-1), kde_func=kde_scipy, gridsize=1000)

			# AUTOCORRELATION FUNCTION
			acfapprox = acf(predicted_traj[:,0], fft=True, nlags=nlags) #look at first component
			acftrue = acf(true_traj[:,0], fft=True, nlags=nlags) #look at first component
			acferr = np.mean((acftrue - acfapprox)**2)


			data = [{"xgrid": xgrid,
						"P_approx": Papprox,
						"P_true": Ptrue,
						"acf_grid": acfgrid,
						"acf_approx": acfapprox,
						"acf_true": acftrue,
						"acf_err": acferr
					}]

			fig_path = os.path.join(self.fig_dir, "kde_{}.png".format(set_name))
			fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 12))
			sns.kdeplot(true_traj.reshape(-1), ax=ax, label='True', linewidth=2)
			sns.kdeplot(predicted_traj.reshape(-1), ax=ax, label='Predicted', linewidth=2)
			ax.set_ylabel("Probability", fontsize=12)
			ax.legend()
			plt.suptitle('KDE with KL-Divergence={kl:.5}'.format(kl=eval_dict['kl_all']))
			plt.savefig(fig_path)
			plt.close()

			fig_path = os.path.join(self.fig_dir, "acf_{}.png".format(set_name))
			fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 12))
			ax.plot(acfgrid, acftrue, label='True', linewidth=2)
			ax.plot(acfgrid, acfapprox, label='Predicted', linewidth=2)
			ax.set_ylabel("ACF", fontsize=12)
			ax.legend()
			plt.suptitle('ACF with error={acf:.5}'.format(acf=acferr))
			plt.savefig(fig_path)
			plt.close()


		self.write_stats(data, "kde_data_{}".format(set_name))

	def testingOnTrainingSet(self):
		with open(self.train_data_path, "rb") as file:
			data = pickle.load(file)
			self.dt_rawdata = data["dt"]
			train_input_sequence = self.subsample(x=data["u_train"][self.trainNumber, :, :self.input_dim], t_end=self.t_test)
		# rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred, hidden_all = self.eval(train_input_sequence, dt, "TRAIN")
		eval_dict = self.eval(input_sequence=train_input_sequence, t_end=self.t_test, set_name="TRAIN"+str(self.trainNumber))
		return eval_dict

	def write_stats(self, pd_stat, stat_name):
		save_path = self.saving_path  + "/{}.pickle".format(stat_name)
		with open(save_path, "wb") as file:
			pickle.dump(pd_stat, file, pickle.HIGHEST_PROTOCOL)

	def subsample(self, x, t_end, dt_given=None, dt_subsample=None):
		if dt_given is None:
			dt_given = self.dt_rawdata

		if dt_subsample is None:
			dt_subsample = self.dt
		# x: time x dims
		n_stop = int(t_end / dt_given) + 1
		keep_inds = [int(j) for j in np.arange(0, n_stop, dt_subsample / dt_given)]
		x_sub = x[keep_inds]
		return x_sub


	### TRAINING STUFF
	def set_random_weights(self):
		# initialize markovian random terms for Random Feature Maps
		self.b_h_markov = np.random.uniform(low=-self.rf_bias_bound, high=self.rf_bias_bound, size=(self.rfDim, 1))
		self.W_in_markov = np.random.uniform(low=-self.rf_Win_bound, high=self.rf_Win_bound, size=(self.rfDim, self.input_dim_rf))

	def x_t(self, t, t0=0):
		# interpolate self.x_vec at time t
		if self.interp=='Linear':
			x = linear_interp(x_vec=self.x_vec, n_min=self.n_min, t=t, t0=t0, dt=self.dt)
		elif self.interp=='Spline':
			x = np.zeros(self.input_dim)
			for k in range(self.input_dim):
				x[k] = self.x_spline[k](t)
		else:
			raise ValueError('Interpolation method for x_t not recognized.')

		return x


	def xdot_t(self, t):
		'''differentiate self.x_vec at time t using stored component-wise spline interpolant'''

		if self.diff =='Euler':
			xdot = linear_interp(x_vec=self.xdot_vec, n_min=self.n_min, t=t, t0=0, dt=self.dt)
		elif self.diff=='TrueDeriv':
			xdot = self.scaler.scaleXdot(self.fTRUE(t=t, y=self.scaler.descaleData(self.x_t(t=t))))
		elif self.diff=='Spline':
			xdot = np.zeros(self.input_dim)
			for k in range(self.input_dim):
				xdot[k] = self.xdot_spline[k](t)
		else:
			raise ValueError('{} not a recognized differentiation method'.format(self.diff))
		return xdot

	def mscaled(self, t, x, xdot):
		'''Assume that x, xdot are both in normalized coordinates'''
		if self.doResidual:
			m = xdot - self.scaler.scaleXdot(self.f0(t, self.scaler.descaleData(x)))
		else:
			m = xdot
		return self.scaler.scaleM(m, reuse=True)

	def q_t(self, x_t):
		q = np.tanh(self.W_in_markov @ x_t + np.squeeze(self.b_h_markov))
		return q

	def qm_t(self, t, k=None):
		x = self.x_t(t=t)
		xdot = self.xdot_t(t=t)
		m = self.mscaled(t, x, xdot)
		if self.component_wise:
			x = x[k,None]
			xdot = xdot[k,None]
			m = m[k,None]

		if self.rf_error_input:
			f0 = self.scaler.scaleXdot(self.f0(t, self.scaler.descaleData(x)))
			if self.component_wise:
				f0 = f0[k,None]
			rf_input = np.hstack((x,f0))
		else:
			rf_input = x

		q = self.q_t(rf_input)

		return q, m

	def Z_t(self, t):
		q, m = self.qm_t(t)
		val = np.outer(q, q).reshape(-1)
		return val

	def Y_t(self, t):
		q, m = self.qm_t(t)
		val = np.outer(q, m)
		return val

	def rcrf_rhs(self, t, S, k=None):
		'''k is the component when doing component-wise models'''
		q, m = self.qm_t(t=t, k=k)
		dZqq = np.outer(q, q).reshape(-1)
		dYq = np.outer(q, m).reshape(-1)
		S = np.hstack( (dZqq, dYq) )
		return S

	def continuousInterpRF(self):
		self.n_min = self.x_vec.shape[0]-1

		if self.doResidual:
			f0_vec = np.array([self.f0(0, x) for x in self.x_vec_raw])
			# xdot(t) = f0(x(t)) + m(t)
			# so, m(t) = xdot(t) - f0(x(t))
			m_vec = self.xdot_vec - self.scaler.scaleXdot(f0_vec)
			self.scaler.scaleM(m_vec) # just stores statistics
		else:
			self.scaler.scaleM(self.xdot_vec) # just stores statistics

		# T_warmup = self.dt*dynamics_length
		T_warmup = 0
		T_train = self.tTrain
		t_span = [T_warmup, T_warmup + T_train]

		if self.component_wise:
			self.Y = []
			self.Z = []
			for k in range(self.input_dim):
				# allocate, reshape, normalize, and save solutions
				self.getYZ(t_span=t_span, k=k)
			self.Y = np.vstack(self.Y)
			self.Z = np.vstack(self.Z)
		else:
			# allocate, reshape, normalize, and save solutions
			self.getYZ(t_span=t_span)

		print('|Z| =', np.mean(self.Z**2))
		print('|Y| =', np.mean(self.Y**2))


	def getYZ(self, t_span, k=None):
		timer_start = time.time()
		if self.component_wise:
			y0 = np.zeros(self.rfDim**2 + self.rfDim)
		else:
			y0 = np.zeros(self.rfDim**2 + self.rfDim*self.input_dim)

		t_eval = np.array([t_span[-1]])
		if self.costIntegrator=='quadvec':
			print('Starting ZY integration with quad_vec...')
			self.newMethod_getYZ_quad(t_span=t_span, k=k)
		else:
			print('Starting ZY integration with solve_ivp...')
			ysol = my_solve_ivp(f_rhs=lambda t, y: self.rcrf_rhs(t, y, k=k), t_span=t_span, t_eval=t_eval, ic=y0, settings=self.solver_settings)
			T_train = t_span[-1] - t_span[0]
			self.newMethod_saveYZ(yend=ysol.T, T_train=T_train)
		print('...took {:2.2f} minutes'.format((time.time() - timer_start)/60))


	def newMethod_getYZ_quad(self, t_span, k=None):
		T = t_span[-1] - t_span[0]

		Z, Zerr = quad_vec(f=self.Z_t, a=t_span[0], b=t_span[-1], limit=self.quad_limit)
		Y, Yerr = quad_vec(f=self.Y_t, a=t_span[0], b=t_span[-1], limit=self.quad_limit)
		self.Z = Z.reshape(self.rfDim, self.rfDim) / T
		self.Y = Y.reshape(self.rfDim, self.input_dim) / T

		self.reg_dim = self.Z.shape[0]

	def newMethod_saveYZ(self, yend, T_train):

		if self.component_wise:
			in_dim = 1
		else:
			in_dim = self.input_dim

		Zqq = yend[:self.rfDim**2]
		Yq = yend[self.rfDim**2:]
		Z = Zqq.reshape(self.rfDim, self.rfDim)
		Y = Yq.reshape(self.rfDim, in_dim)

		# save Time-Normalized Y,Z
		if self.component_wise:
			self.Y.append(Y / T_train)
			self.Z.append(Z / T_train)
		else:
			self.Y = Y / T_train
			self.Z = Z / T_train

		# store Z size for building regularization identity matrix
		self.reg_dim = Z.shape[0]


	def doNewSolving(self, do_plots=True):
		if 'GP' not in self.modelType:
			print('Solving inverse problem W = (Z+rI)^-1 Y...')
			# regI = np.identity(self.Z.shape[0])
			regI = np.identity(self.reg_dim)
			regI *= self.regularization_RF

			if self.component_wise:
				# stack regI K times
				regI = np.tile(regI,(self.input_dim,1))

			pinv_ = scipypinv2(self.Z + regI)
			W_out_all = (pinv_ @ self.Y).T
			self.W_out_markov = W_out_all

			if do_plots:
				plotMatrix(self, self.W_out_markov, 'W_out_markov')
				self.plotModel()

			# Compute residuals from inversion
			res = (self.Z + regI) @ W_out_all.T - self.Y
			mse = np.mean(res**2)
			print('Inversion MSE for lambda_RF={lrf} is {mse} with normalized |Wout|={nrm}'.format(lrf=self.regularization_RF, mse=mse, nrm=np.mean(W_out_all**2)))

	def get_regression_IO(self):

		if 'Psi' in self.modelType:
			x_input = np.copy(self.x_vec[:-1])
			x_output = np.copy(self.x_vec[1:])
			x_input_descaled = self.scaler.descaleData(x_input)
		elif 'rhs' in self.modelType:
			x_input = np.copy(self.x_vec)
			x_output = np.copy(self.xdot_vec)
			x_input_descaled = self.scaler.descaleData(x_input)
		else:
			raise ValueError('Did not reecognize self.modelType')

		rf_input, x_output, x_input_descaled = self.get_regIO(x_input=x_input, x_output=x_output)

		return rf_input, x_output, x_input_descaled


	def get_regIO(self, x_input, x_output=None):
		x_input_descaled = self.scaler.descaleData(x_input)

		if self.usef0:
			if 'Psi' in self.modelType:
				predf0 = self.scaler.scaleData(np.array([self.psi0(x_input_descaled[i]) for i in range(x_input.shape[0])]), reuse=1)
			elif 'rhs' in self.modelType:
				predf0 = self.scaler.scaleXdot(np.array([self.f0(0, x_input_descaled[i]) for i in range(x_input.shape[0])]))
			else:
				raise ValueError('Did not reecognize self.modelType')

			# subtract residual
			if self.doResidual and (x_output is not None):
				x_output -= predf0

		if self.component_wise:
			rf_input = []
			for k in range(self.input_dim):
				if self.rf_error_input:
					rf_input_k = np.hstack((x_input[:,k,None], predf0[:,k,None]))
				else:
					rf_input_k = x_input[:,k,None]
				rf_input.append(rf_input_k)
			rf_input = np.array(rf_input)
			if x_output is not None:
				x_output = x_output.T
		else:
			if self.rf_error_input:
				rf_input = np.hstack((x_input, predf0))
			else:
				rf_input = x_input

		return rf_input, x_output, x_input_descaled

	def plotModel(self):

		if not self.component_wise:
			return

		rf_input, x_output, x_input_descaled = self.get_regression_IO()

		x_grid = np.arange(-8,13,0.01)
		x_grid_scaled_mat = self.scaler.scaleData(x_grid, reuse=1)[None,:]
		bh_mat = np.tile(self.b_h_markov, len(x_grid))
		hY = self.W_out_markov @ np.tanh(self.W_in_markov @ x_grid_scaled_mat + bh_mat)
		if 'Psi' in self.modelType:
			x_output /= self.dt
			hY /= self.dt

		fig_path = os.path.join(self.fig_dir, "model_plot.png")
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

		ax.plot(x_grid, np.squeeze(hY), color='blue', linewidth=4)
		ax.scatter(x_input_descaled.reshape(-1), x_output.T.reshape(-1))
		ax.set_xlabel('X_k')
		ax.set_ylabel('hY')
		plt.suptitle('Model Fit')
		plt.savefig(fig_path)
		plt.close()

	def saveModel(self):
		# print("Recording time...")
		self.total_training_time = time.time() - self.start_time
		print("Total training time is {:2.2f} minutes".format(self.total_training_time/60))

		# print("MEMORY TRACKING IN MB...")
		# process = psutil.Process(os.getpid())
		# memory = process.memory_info().rss/1024/1024
		# self.memory = memory
		# print("Script used {:} MB".format(self.memory))
		if self.f0only:
			data = {
			"scaler":self.scaler
			}
		elif 'GP' not in self.modelType:
			self.n_trainable_parameters = np.size(self.W_out_markov)
			self.n_model_parameters = np.size(self.W_in_markov) + np.size(self.b_h_markov)
			self.n_model_parameters += self.n_trainable_parameters
			print("Number of trainable parameters: {}".format(self.n_trainable_parameters))
			print("Total number of parameters: {}".format(self.n_model_parameters))
			data = {
			"differentiation_error":self.differentiation_error,
			"n_trainable_parameters":self.n_trainable_parameters,
			"n_model_parameters":self.n_model_parameters,
			"total_training_time":self.total_training_time,
			"W_in_markov":self.W_in_markov,
			"b_h_markov":self.b_h_markov,
			"W_out_markov":self.W_out_markov,
			"scaler":self.scaler,
			"regularization_RF":self.regularization_RF,
			"rf_Win_bound":self.rf_Win_bound,
			"rf_bias_bound":self.rf_bias_bound
			}
		elif 'GP' in self.modelType:
			data = {
			"differentiation_error":self.differentiation_error,
			"scaler":self.scaler,
			"total_training_time":self.total_training_time,
			"gpr": self.gpr
			}
		data_path = os.path.join(self.model_dir, "data.pickle")
		with open(data_path, "wb") as file:
			pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
			del data
		return 0

	def loadModel(self):
		data_path = os.path.join(self.model_dir, "data.pickle")
		with open(data_path, "rb") as file:
			data = pickle.load(file)
			self.scaler = data["scaler"]
			if not self.f0only:
				self.W_in_markov = data["W_in_markov"]
				self.b_h_markov = data["b_h_markov"]
				self.W_out_markov = data["W_out_markov"]
				self.regularization_RF = data["regularization_RF"]
				self.rf_Win_bound = data["rf_Win_bound"]
				self.rf_bias_bound = data["rf_bias_bound"]
				self.differentiation_error = data["differentiation_error"]
