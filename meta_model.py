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

# from scipy.stats import loguniform
import time
from functools import partial
print = partial(print, flush=True)
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from odelibrary import my_solve_ivp, L63

# matt utils
from utils import file_to_dict
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

		self.scaler = scaler(tt=self.scaler_tt, tt_derivative=self.scaler_tt_derivatives, component_wise=self.component_wise)
		self.fig_dir = os.path.join(self.saving_path, params["fig_dir"])
		self.model_dir = os.path.join(self.saving_path, params["model_dir"])
		self.logfile_dir = os.path.join(self.saving_path, params["logfile_dir"])
		np.random.seed(self.rng_seed)
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
		self.set_fidelity('medfi')
		if self.validate_hyperparameters:
			# switch to lowfi quadrature for cheap validation runs
			self.set_fidelity('lowfi')

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
			# return to hifi quadrature for final solve
			self.set_fidelity('medfi')
			self.setup_the_learning()

			# now fix the optimal W,b and the learned Y,Z... just learn the optimal regularization for the inversion
			self.set_fidelity('lowfi')
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
			self.set_fidelity('medfi')
		else:
			self.setup_the_learning()

		# solve for the final Y,Z, regI and save
		self.doNewSolving()
		self.saveModel()

	def setup_timeseries(self):
		self.xdot_vec_TRUE = np.copy(self.train_input_sequence_dot)
		self.x_vec = np.copy(self.train_input_sequence)
		self.x_vec_raw = self.scaler.descaleData(self.x_vec)
		t_vec = self.dt*np.arange(self.x_vec.shape[0])
		# get derivative
		if self.diff == 'TrueDeriv': # use true derivative
			self.xdot_vec = np.copy(self.xdot_vec_TRUE)
		elif self.diff == 'InterpThenDiff': # do spline derivativer
			self.xdot_spline = [CubicSpline(x=t_vec, y=self.x_vec[:,k]).derivative() for k in range(self.input_dim)]
			# get m(t) for all time JUST to have its statistics for normalization ahead of time
			self.xdot_vec = np.array([self.xdot_spline[k](t_vec) for k in range(self.input_dim)]).T
		elif self.diff == 'DiffThenInterp': # do spline derivativer
			self.xdot_vec = np.zeros(self.xdot_vec_TRUE.shape)
			for k in range(len(t_vec)-1):
				self.xdot_vec[k] = (self.x_vec[k+1] - self.x_vec[k]) / self.dt
			# need to deal with boundary issue
			self.xdot_vec[-1] = self.xdot_vec[-2]
		else:
			print('Not differentiating.')
			self.differentiation_error = np.nan
			return

		self.differentiation_error = np.mean((self.xdot_vec-self.xdot_vec_TRUE)**2)
		print('|XdotInferred - XdotTRUE|=', self.differentiation_error)


	def setup_the_learning(self):
		tl = self.x_vec.shape[0] - self.dynamics_length

		if 'continuous' in self.modelType:
			## Random WEIGHTS
			self.set_random_weights()
			self.continuousInterpRF()
		elif 'Euler' in self.modelType:
			if self.component_wise:
				raise ValueError('component wise not yet set up for Euler')
			x_input = np.copy(self.x_vec)
			x_output = np.copy(self.xdot_vec)
			x_input_descaled = self.scaler.descaleData(x_input)

			if self.usef0:
				predf0 = self.scaler.scaleXdot(np.array([self.f0(0, x_input_descaled[i]) for i in range(x_input.shape[0])]))
			if self.rf_error_input:
				rf_input = np.hstack((x_input, predf0))
			else:
				rf_input = x_input
			if self.doResidual:
				x_output -= predf0

			if 'GP' in self.modelType:
				GP_ker = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e+6)) + WhiteKernel(1.0, (1e-10, 1e6))
				my_gpr = GaussianProcessRegressor(
					kernel = GP_ker,
					n_restarts_optimizer = 15,
					alpha = 1e-10)
				self.gpr = my_gpr.fit(X=rf_input, y=x_output)
			else:
				self.set_random_weights()
				Q = np.array([self.q_t(rf_input[i]) for i in range(rf_input.shape[0])])
				self.Z = Q.T @ Q / x_input.shape[0] # normalize by length
				self.Y = Q.T @ x_output / x_input.shape[0]
				self.reg_dim = self.Z.shape[0]
				print('|Z| =', np.mean(self.Z**2))
				print('|Y| =', np.mean(self.Y**2))

		elif 'discrete' in self.modelType:
			if self.component_wise:
				raise ValueError('component wise not yet set up for discrete')
			x_output = np.copy(self.x_vec[1:])
			x_input = np.copy(self.x_vec[:-1])
			x_input_descaled = self.scaler.descaleData(x_input)

			if self.usef0:
				predf0 = self.scaler.scaleData(np.array([self.psi0(x_input_descaled[i]) for i in range(x_input.shape[0])]), reuse=1)
			if self.rf_error_input:
				rf_input = np.hstack((x_input, predf0))
			else:
				rf_input = x_input
			if self.doResidual:
				x_output -= predf0

			if self.modelType=='discrete':
				self.set_random_weights()
				Q = np.array([self.q_t(rf_input[i]) for i in range(rf_input.shape[0])])
				self.Z = Q.T @ Q / x_input.shape[0] # normalize by length
				self.Y = Q.T @ x_output / x_input.shape[0]
				self.reg_dim = self.Z.shape[0]
				print('|Z| =', np.mean(self.Z**2))
				print('|Y| =', np.mean(self.Y**2))
			elif self.modelType=='discreteGP':
				GP_ker = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e+6)) + WhiteKernel(1.0, (1e-10, 1e6))
				my_gpr = GaussianProcessRegressor(
					kernel = GP_ker,
					n_restarts_optimizer = 15,
					alpha = 1e-10)
				self.gpr = my_gpr.fit(X=rf_input, y=x_output)

		# Store something useful for plotting
		# self.first_train_vec = train_input_sequence[(self.dynamics_length+1),:]

	def test(self):
		for fidelity in ['Euler', 'default', 'lowfi', 'medfi', 'hifi', 'hifiPlus']:
			self.set_fidelity(fidelity)
			test_eval = self.testingOnSet(setnm='test', fidelity_name=fidelity)
			self.write_stats(pd_stat=test_eval, stat_name='test_eval_{}'.format(fidelity))
			if fidelity=='hifi':
				self.write_stats(pd_stat=test_eval, stat_name='test_eval')
			print(test_eval.mean())

	def validate(self):
		# self.testingOnTrainingSet()
		validate_eval = self.testingOnSet(setnm='validate', do_plots=False)
		# self.saveResults()
		# self.write_testing_stats()
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
		quality = quality_df.t_valid_050.mean()
		return quality


	def testingOnSet(self, setnm, do_plots=True, fidelity_name=''):
		with open(self.test_data_path, "rb") as file:
			data = pickle.load(file)
			self.dt_rawdata = data["dt"]
			eval_data = data["u_{}".format(setnm)][:, :, :self.input_dim]
			del data

		n_traj = eval_data.shape[0]
		test_eval = []
		# loop over test sets
		for n in range(n_traj):
			print('Evaluating',setnm, 'set #', n+1, '/', n_traj)
			test_input_sequence = self.subsample(x=eval_data[n], t_end=self.t_test)
			eval_dict = self.eval(input_sequence=test_input_sequence, t_end=self.t_test, set_name=setnm+fidelity_name+str(n), do_plots=do_plots)
			test_eval.append(eval_dict)

		# regroup test_eval
		test_eval = pd.DataFrame(test_eval)

		return test_eval
		# rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred, hidden_all = self.eval(test_input_sequence, dt, "TEST")

	def eval(self, input_sequence, t_end, set_name, do_plots=True):
		# allocate ic and target
		ic = self.scaler.scaleData(input_sequence[0], reuse=1)
		target = input_sequence

		# get predictions
		prediction = self.make_predictions(ic=ic, t_end=t_end)
		prediction = self.scaler.descaleData(prediction)
		eval_dict = computeErrors(target, prediction, self.scaler.data_std, dt=self.dt)

		# add hyperparameters to test_eval.pickle output
		try:
			for my_varnm in ["regularization_RF", "rf_Win_bound", "rf_bias_bound"]:
				exec("val = self.{varnm}".format(varnm=my_varnm))
				eval_dict[my_varnm] = val
		except:
			pass

		if do_plots:
			self.makeNewPlots(true_traj=target, predicted_traj=prediction, set_name=set_name)
		return eval_dict

	def make_predictions(self, ic, t_end):
		if 'discrete' in self.modelType or self.f0only:
			prediction = []
			prediction.append(ic)
			n_steps = int(t_end / self.dt)
			for n in range(n_steps):
				ic = self.predict_next(x_input=ic)
				prediction.append(ic)
			prediction = np.array(prediction)
		elif ('continuous' in self.modelType) or ('Euler' in self.modelType):
			# if 'Euler' in self.modelType:
			# 	self.solver_settings['method'] = 'Euler'
			# 	self.solver_settings['dt'] = self.dt

			N = int(t_end / self.dt) + 1
			t_eval = self.dt*np.arange(N)
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
		if 'discrete' in self.modelType:
			if self.doResidual or self.usef0:
				pred = self.scaler.scaleData(self.psi0(self.scaler.descaleData(x_input)), reuse=1)
			else:
				pred = 0
			if self.rf_error_input:
				rf_input = np.hstack((x_input, pred))
			else:
				rf_input = x_input
			if 'GP' in self.modelType:
				u_next = pred + self.gpr.predict(rf_input)
			else:
				u_next = pred + self.W_out_markov @ self.q_t(rf_input)
		elif 'continuous' in self.modelType:
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

		# add mechanistic rhs
		if self.usef0:
			f0 = self.scaler.scaleXdot(self.f0(t0, self.scaler.descaleData(x_input)))

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

	def makeNewPlots(self, true_traj, predicted_traj, set_name=''):
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

	def subsample(self, x, t_end):
		# x: time x dims
		n_stop = int(t_end / self.dt_rawdata) + 1
		keep_inds = [int(j) for j in np.arange(0, n_stop, self.dt / self.dt_rawdata)]
		x_sub = x[keep_inds]
		return x_sub


	### TRAINING STUFF
	def set_random_weights(self):
		# initialize markovian random terms for Random Feature Maps
		self.b_h_markov = np.random.uniform(low=-self.rf_bias_bound, high=self.rf_bias_bound, size=(self.rfDim, 1))
		self.W_in_markov = np.random.uniform(low=-self.rf_Win_bound, high=self.rf_Win_bound, size=(self.rfDim, self.input_dim_rf))

	def x_t(self, t, t0=0):
		#linearly interpolate self.x_vec at time t
		return linear_interp(x_vec=self.x_vec, n_min=self.n_min, t=t, t0=t0, dt=self.dt)

	def xdot_t(self, t):
		'''differentiate self.x_vec at time t using stored component-wise spline interpolant'''

		if self.diff =='DiffThenInterp':
			xdot = linear_interp(x_vec=self.xdot_vec, n_min=self.n_min, t=t, t0=0, dt=self.dt)
		elif self.diff=='TrueDeriv':
			xdot = self.fTRUE(t=t, y=self.x_t(t=t))
		elif self.diff=='InterpThenDiff':
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
		step = self.dt/10
		t_eval = np.array([t_span[-1]])
		y0 = self.newMethod_getIC(T_warmup=T_warmup)

		if self.component_wise:
			self.Y = []
			self.Z = []
			for k in range(self.input_dim):
				# allocate, reshape, normalize, and save solutions
				print('Compute final Y,Z component...')
				if self.ZY=='new':
					self.newMethod_getYZ_quad(t_span=t_span, k=k)
				else:
					print('Integrating over training data...')
					timer_start = time.time()
					ysol = my_solve_ivp(f_rhs=lambda t, y: self.rcrf_rhs(t, y, k=k), t_span=t_span, t_eval=t_eval, ic=y0[k], settings=self.solver_settings)
					print('...took {:2.2f} minutes'.format((time.time() - timer_start)/60))
					self.newMethod_saveYZ(yend=ysol.T, T_train=T_train)
			self.Y = np.vstack(self.Y)
			self.Z = np.vstack(self.Z)
		else:
			# allocate, reshape, normalize, and save solutions
			print('Computing final Y,Z...')
			if self.ZY=='new':
				print('Integrating Z, Y with quadrature')
				timer_start = time.time()
				self.newMethod_getYZ_quad(t_span=t_span)
				print('...took {:2.2f} minutes'.format((time.time() - timer_start)/60))
			else:
				print('Integrating Z, Y with ODE solver')
				timer_start = time.time()
				ysol = my_solve_ivp(f_rhs=self.rcrf_rhs, t_span=t_span, t_eval=t_eval, ic=y0, settings=self.solver_settings)
				print('...took {:2.2f} minutes'.format((time.time() - timer_start)/60))
				self.newMethod_saveYZ(yend=ysol.T, T_train=T_train)

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

			# Compute residuals from inversion
			res = (self.Z + regI) @ W_out_all.T - self.Y
			mse = np.mean(res**2)
			print('Inversion MSE for lambda_RF={lrf} is {mse} with normalized |Wout|={nrm}'.format(lrf=self.regularization_RF, mse=mse, nrm=np.mean(W_out_all**2)))

	def newMethod_getIC(self, T_warmup):
		# generate ICs for training integration
		yall = []

		if self.ZY=='old':
			x0 = self.x_t(t=T_warmup)
			xdot0 = self.xdot_t(t=T_warmup)
			m0 = self.mscaled(t=T_warmup, x=x0, xdot=xdot0)
			if self.rf_error_input:
				f0 = self.scaler.scaleXdot(self.f0(T_warmup, self.scaler.descaleData(x0)))
			if self.component_wise:
				for k in range(self.input_dim):
					x0k = x0[k,None]
					m0k = m0[k,None]
					if self.rf_error_input:
						f0k = f0[k,None]
						rf_input = np.hstack((x0k,f0k))
					else:
						rf_input = x0k
					q0k = self.q_t(rf_input)
					Zqq0 = np.outer(q0k, q0k).reshape(-1)
					Yq0 = np.outer(q0k, m0k).reshape(-1)
					y0 = np.hstack( (Zqq0, Yq0) )
					yall.append(y0)
			else:
				if self.rf_error_input:
					rf_input = np.hstack((x0,f0))
				else:
					rf_input = x0
				q0 = self.q_t(rf_input)
				Zqq0 = np.outer(q0, q0).reshape(-1)
				Yq0 = np.outer(q0, m0).reshape(-1)
				yall = np.hstack( (Zqq0, Yq0) )

		yall = np.array(yall)

		return yall


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
