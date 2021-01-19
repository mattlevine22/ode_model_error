import os
import numpy as np
import pickle
from scipy import signal # for periodogram
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp, trapz
from scipy.interpolate import CubicSpline
# from scipy.stats import loguniform
from esn_plotting_utils import *
from esn_global_utils import *
import time
from functools import partial
print = partial(print, flush=True)
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from new_utils import file_to_dict
from odelibrary import L63

import pandas as pd

#memory tracking
# import psutil


import pdb

class IDK(object):
	def __init__(self, settings, default_esn_settings='./Config/esn_default_params.json'):
		# load default settings
		params = file_to_dict(default_esn_settings)

		# add extra settings and replace defaults as needed
		params.update(settings)

		# initialize
		## new properties
		self.modelType = params["modelType"]
		self.stateType = params["stateType"]
		self.trainNumber = params["trainNumber"]
		self.dt = params["dt"]
		self.t_train = params["tTrain"]
		self.t_test = params["t_test"]

		self.f0only = 0
		if self.modelType=='f0only':
			self.f0only = 1

		## set paths
		self.train_data_path = params["train_data_path"]
		self.test_data_path = params["test_data_path"]
		self.fig_dir = params["fig_dir"]
		self.model_dir = params["model_dir"]
		self.logfile_dir = params["logfile_dir"]
		self.results_dir = params["results_dir"]
		self.saving_path = params["saving_path"]

		self.rng_seed = params["rng_seed"]
		np.random.seed(self.rng_seed)

		self.input_dim = params["input_dim"]
		self.regularization_RF = params["regularization_RF"]
		self.scaler_tt = params["scaler"]
		self.scaler_tt_derivatives = params["scaler_derivatives"]

		self.dynamics_length = 1

		##########################################
		self.component_wise = params["component_wise"]
		self.scaler = scaler(tt=self.scaler_tt, tt_derivative=self.scaler_tt_derivatives, component_wise=self.component_wise)
		self.noise_level = params["noise_level"]
		self.test_integrator = params["test_integrator"]
		self.rf_dim = params["rfDim"]
		self.rf_Win_bound = params["rf_Win_bound"]
		self.rf_bias_bound = params["rf_bias_bound"]
		self.ZY = params["ZY"]

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

		os.makedirs(self.saving_path + self.model_dir, exist_ok=True)
		os.makedirs(self.saving_path + self.fig_dir, exist_ok=True)
		os.makedirs(self.saving_path + self.results_dir, exist_ok=True)
		os.makedirs(self.saving_path + self.logfile_dir, exist_ok=True)

		print('FIGURE PATH:', self.saving_path + self.results_dir)


	def train(self):
		# if self.dont_redo and os.path.exists(self.saving_path + self.model_dir + "/data.pickle"):
		# 	raise ValueError('Model has already run for this configuration. Exiting with an error.')

		self.start_time = time.time()

		## DATA
		with open(self.train_data_path, "rb") as file:
			# Pickle the "data" dictionary using the highest protocol available.
			data = pickle.load(file)
			self.dt_rawdata = data["dt"]
			train_input_sequence = self.subsample(x=data["u_train"][self.trainNumber, :, :self.input_dim], t_end=self.t_train)
			del data
		# scale training data
		train_input_sequence = self.scaler.scaleData(train_input_sequence)

		if self.f0only:
			# need to normalize first to store statistics
			return

		## Random WEIGHTS
		self.set_random_weights()

		# TRAINING LENGTH
		tl = train_input_sequence.shape[0] - self.dynamics_length

		# Do the learning!
		self.setup_the_learning(tl=tl, dynamics_length=self.dynamics_length, train_input_sequence=train_input_sequence)

		# get answers for default hyperparameters (regularization_RF)
		# self.doNewSolving()
		# self.saveModel()

		# Do validation loop over reg_RF
		# reg_list = [10, 5, 1, 5e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
		reg_list = [10, 1, 0.1, 0.01, 1e-3, 1e-6, 1e-9]
		# reg_list = [1, 10, 1e-6]
		best_reg = self.regularization_RF
		best_quality = 0
		for reg_rf in reg_list:
			self.regularization_RF = reg_rf
			self.doNewSolving()
			quality_df = self.validate()
			quality = quality_df.t_valid_050.mean()
			if quality > best_quality:
				best_quality = quality
				best_reg = reg_rf
				print('Best: ', best_reg, 'with t_valid=', best_quality)
				self.saveModel()

		# run final
		# self.regularization_RF = best_reg
		# self.doNewSolving()
		# self.saveModel()

		# output training statistics
		self.write_training_stats()

	def setup_the_learning(self, tl, dynamics_length, train_input_sequence):
		if self.modelType=='continuousInterp':
			self.continuousInterpRF(tl=tl, dynamics_length=self.dynamics_length, train_input_sequence=train_input_sequence)
		elif self.modelType=='discrete':
			if self.component_wise:
				raise ValueError('component wise not yet set up for discrete')
			x_output = train_input_sequence[1:]
			x_input = train_input_sequence[:-1]
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
				Z = np.zeros((self.rf_dim, self.rf_dim))
				Y = np.zeros((self.rf_dim, self.input_dim))
				for i in range(x_input.shape[0]):
					q_i = self.q_t(rf_input[i])
					foo = x_output[i]
					Z += np.outer(q_i, q_i)
					Y += np.outer(q_i, foo)
				self.Z = Z / x_input.shape[0]
				self.Y = Y / x_input.shape[0]
				self.reg_dim = self.Z.shape[0]
			elif self.modelType=='discreteGP':
				return

		# Store something useful for plotting
		# self.first_train_vec = train_input_sequence[(self.dynamics_length+1),:]

	def test(self):
		# self.testingOnTrainingSet()
		test_eval = self.testingOnSet(setnm='test')
		# self.saveResults()
		self.write_stats(pd_stat=test_eval, stat_name='test_eval')

	def validate(self):
		# self.testingOnTrainingSet()
		validate_eval = self.testingOnSet(setnm='validate', do_plots=False)
		# self.saveResults()
		# self.write_testing_stats()
		return validate_eval


	def testingOnSet(self, setnm, do_plots=True):
		with open(self.test_data_path, "rb") as file:
			data = pickle.load(file)
			self.dt_rawdata = data["dt"]
			test_data = data["u_{}".format(setnm)][:, :, :self.input_dim]
			del data

		n_test_traj = test_data.shape[0]
		test_eval = []
		# loop over test sets
		for n in range(n_test_traj):
			test_input_sequence = self.subsample(x=test_data[n], t_end=self.t_test)
			eval_dict = self.eval(input_sequence=test_input_sequence, t_end=self.t_test, set_name=setnm+str(n), do_plots=do_plots)
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
		if do_plots:
			self.makeNewPlots(true_traj=target, predicted_traj=prediction, set_name=set_name)
		return eval_dict

	def make_predictions(self, ic, t_end):
		prediction = []
		n_steps = int(t_end / self.dt)
		for n in range(n_steps):
			# t = n*dt
			prediction.append(ic)
			ic = self.predict_next(x_input=ic)
		prediction = np.array(prediction)
		return prediction


	def psi0(self, ic):
		pred = self.forward_solver(x_input=ic, f_rhs=self.f0)
		return pred

	def predict_next(self, x_input, t0=0):
		u_next = np.zeros(self.input_dim)
		if self.modelType=='discrete':
			if self.doResidual or self.usef0:
				pred = self.scaler.scaleData(self.psi0(self.scaler.descaleData(x_input)), reuse=1)
			else:
				pred = 0
			if self.rf_error_input:
				rf_input = np.hstack((x_input, pred))
			else:
				rf_input = x_input
			u_next = pred + self.W_out_markov @ self.q_t(rf_input)
		elif self.modelType=='continuousInterp':
			u_next = self.forward_solver(x_input=x_input, f_rhs=self.rhs)
		elif self.modelType=='f0only':
			u_next = self.scaler.scaleData(self.psi0(self.scaler.descaleData(x_input)), reuse=1)
		else:
			raise ValueError('modelType not recognized')

		return np.squeeze(u_next)


	def forward_solver(self, x_input, f_rhs, t0=0):
		solver = self.test_integrator
		u0 = np.copy(x_input)
		if solver=='Euler':
			rhs = f_rhs(t0, u0)
			u_next = u0 + self.dt * rhs
		elif solver=='Euler_fast':
			dt_fast = self.dt_fast_frac * self.dt
			t_end = t0 + self.dt
			t = np.float(t0)
			u_next = np.copy(u0)
			while t < t_end:
				rhs = f_rhs(t, u_next)
				u_next += dt_fast * rhs
				t += dt_fast
		elif solver=='RK45':
			t_span = [t0, t0+self.dt]
			t_eval = np.array([t0+self.dt])
			sol = solve_ivp(fun=lambda t, y: f_rhs(t, y), t_span=t_span, y0=u0, t_eval=t_eval, max_step=self.dt/10)
			u_next = sol.y

		u_next = np.squeeze(u_next)
		return u_next

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
		fig_path = self.saving_path + self.fig_dir + "/timewise_fits_{}.png".format(set_name)
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
		fig_path = self.saving_path + self.fig_dir + "/timewise_errors_{}.png".format(set_name)
		fig, ax = plt.subplots(nrows=self.input_dim, ncols=1,figsize=(12, 12))
		for k in range(self.input_dim):
			ax[k].set_ylabel(r"$Y_{k}$".format(k=k), fontsize=12)
			ax[k].plot(time_vec, true_traj[:,k] - predicted_traj[:,k], linewidth=2)
		plt.suptitle('Timewise errors with total MSE {mse:.5}'.format(mse=mse))
		plt.savefig(fig_path)
		plt.close()

	def loadModel(self):
		data_path = self.saving_path + self.model_dir + "/data.pickle"
		try:
			with open(data_path, "rb") as file:
				data = pickle.load(file)
				self.W_in_markov = data["W_in_markov"]
				self.b_h_markov = data["b_h_markov"]
				self.W_out_markov = data["W_out_markov"]
				self.scaler = data["scaler"]
				# self.first_train_vec = data["first_train_vec"]
			return 0
		except:
			print("MODEL {:s} NOT FOUND.".format(data_path))
			return 1


	def testingOnTrainingSet(self):
		with open(self.train_data_path, "rb") as file:
			data = pickle.load(file)
			self.dt_rawdata = data["dt"]
			train_input_sequence = self.subsample(x=data["u_train"][self.trainNumber, :, :self.input_dim], t_end=self.t_test)
		# rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred, hidden_all = self.eval(train_input_sequence, dt, "TRAIN")
		eval_dict = self.eval(input_sequence=train_input_sequence, t_end=self.t_test, set_name="TRAIN"+str(self.trainNumber))
		return eval_dict

	def plot(self):
		pass

	def write_training_stats(self):
	    pass

	def write_stats(self, pd_stat, stat_name):
		save_path = self.saving_path  + "/{}.pickle".format(stat_name)
		with open(save_path, "wb") as file:
			pickle.dump(pd_stat, file, pickle.HIGHEST_PROTOCOL)

	def subsample(self, x, t_end):
		# x: time x dims
		n_stop = int(t_end / self.dt_rawdata)
		keep_inds = [int(j) for j in np.arange(0, n_stop, self.dt / self.dt_rawdata)]
		x_sub = x[keep_inds]
		return x_sub


	### TRAINING STUFF
	def set_random_weights(self):

		# First initialize everything to be None
		self.W_in_markov = None
		self.b_h_markov = None
		self.W_out_markov = None

		# initialize markovian random terms for Random Feature Maps
		b_h_markov = np.random.uniform(low=-self.rf_bias_bound, high=self.rf_bias_bound, size=(self.rf_dim, 1))
		W_in_markov = np.random.uniform(low=-self.rf_Win_bound, high=self.rf_Win_bound, size=(self.rf_dim, self.input_dim_rf))

		self.W_in_markov = W_in_markov
		self.b_h_markov = b_h_markov

	def x_t(self, t, t0=0):
		#linearly interpolate self.x_vec at time t
		# self.n_min = self.x_vec.shape[0]-1
		ind_mid = (t-t0) / self.dt
		ind_low = max(0, min( int(np.floor(ind_mid)), self.n_min) )
		ind_high = min(self.n_min, int(np.ceil(ind_mid)))
		v0 = self.x_vec[ind_low,:]
		v1 = self.x_vec[ind_high,:]
		tmid = ind_mid - ind_low

		return (1 - tmid) * v0 + tmid * v1

	def xdot_t(self, t):
		'''differentiate self.x_vec at time t using stored component-wise spline interpolant'''

		# initialize output
		xdot = np.zeros(self.input_dim)
		for k in range(self.input_dim):
			xdot[k] = self.xdot_spline[k](t)
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

	def rcrf_rhs(self, t, S, k=None):
		'''k is the component when doing component-wise models'''

		x = self.x_t(t=t)
		xdot = self.xdot_t(t=t)
		m = self.mscaled(t, x, xdot)
		if self.component_wise:
			x = x[k,None]
			xdot = xdot[k,None]
			m = m[k,None]

		if self.rf_error_input:
			rf_input = np.hstack((x,m))
		else:
			rf_input = x

		if self.ZY=='new':
			return 0
		else:
			q = self.q_t(rf_input)
			dZqq = np.outer(q, q).reshape(-1)
			dYq = np.outer(q, m).reshape(-1)
			S = np.hstack( (dZqq, dYq) )
			return S

	def newMethod_getIC(self, T_warmup, T_train):
		# generate ICs for training integration
		yall = []

		if self.ZY=='old':
			x0 = self.x_t(t=T_warmup)
			xdot0 = self.xdot_t(t=T_warmup)
			m0 = self.mscaled(t=T_warmup, x=x0, xdot=xdot0)
			if self.component_wise:
				for k in range(self.input_dim):
					x0k = x0[k,None]
					m0k = m0[k,None]
					if self.rf_error_input:
						rf_input = np.hstack((x0k,m0k))
					else:
						rf_input = x0k
					q0k = self.q_t(rf_input)
					Zqq0 = np.outer(q0k, q0k).reshape(-1)
					Yq0 = np.outer(q0k, m0k).reshape(-1)
					y0 = np.hstack( (Zqq0, Yq0) )
					yall.append(y0)
			else:
				if self.rf_error_input:
					rf_input = np.hstack((x0,m0))
				else:
					rf_input = x0
				q0 = self.q_t(rf_input)
				Zqq0 = np.outer(q0, q0).reshape(-1)
				Yq0 = np.outer(q0, m0).reshape(-1)
				yall = np.hstack( (Zqq0, Yq0) )

		yall = np.array(yall)

		return yall

	def continuousInterpRF(self, tl, dynamics_length, train_input_sequence):

		self.x_vec = train_input_sequence
		self.n_min = self.x_vec.shape[0]-1
		self.x_vec_raw = self.scaler.descaleData(train_input_sequence)

		# get spline derivative
		t_vec = self.dt*np.arange(self.x_vec.shape[0])
		self.xdot_spline = [CubicSpline(x=t_vec, y=self.x_vec[:,k]).derivative() for k in range(self.input_dim)]
		self.xdot_spline_raw = [CubicSpline(x=t_vec, y=self.x_vec_raw[:,k]).derivative() for k in range(self.input_dim)]

		# get m(t) for all time JUST to have its statistics for normalization ahead of time
		xdot_vec = np.array([self.xdot_spline[k](t_vec) for k in range(self.input_dim)]).T
		xdot_vec_raw = np.array([self.xdot_spline_raw[k](t_vec) for k in range(self.input_dim)]).T


		# troubleshoot
		xdot0 = np.array([self.xdot_spline[k](0) for k in range(self.input_dim)]).T
		xdot0_raw = np.array([self.xdot_spline_raw[k](0) for k in range(self.input_dim)]).T

		if self.doResidual:
			f0_vec = np.array([self.f0(0, x) for x in self.scaler.descaleData(self.x_vec)])
			# xdot(t) = f0(x(t)) + m(t)
			# so, m(t) = xdot(t) - f0(x(t))
			m_vec = xdot_vec - self.scaler.scaleXdot(f0_vec)
			self.scaler.scaleM(m_vec) # just stores statistics
		else:
			self.scaler.scaleM(xdot_vec) # just stores statistics


		# T_warmup = self.dt*dynamics_length
		T_warmup = 0
		T_train = self.t_train
		t_span = [T_warmup, T_warmup + T_train]
		step = self.dt/10
		t_eval = np.linspace(start=t_span[0], stop=t_span[-1], num=int(T_train/step))
		y0 = self.newMethod_getIC(T_warmup=T_warmup, T_train=T_train)

		if self.component_wise:
			self.Y = []
			self.Z = []
			for k in range(self.input_dim):
				# Perform training integration using IC y0

				# allocate, reshape, normalize, and save solutions
				print('Compute final Y,Z component...')
				if self.ZY=='new':
					self.newMethod_getYZstuff2(times=t_eval, k=k)
					self.newMethod_saveYZ(T_train=T_train)
				else:
					print('Integrating over training data...')
					timer_start = time.time()
					sol = solve_ivp(fun=lambda t, y: self.rcrf_rhs(t, y, k=k), t_span=t_span, t_eval=t_span, y0=y0[k])
					print('...took {:2.2f} minutes'.format((time.time() - timer_start)/60))
					self.newMethod_getYZstuff(yend=sol.y[:,-1])
					self.newMethod_saveYZ(T_train=T_train)

			self.Y = np.vstack(self.Y)
			self.Z = np.vstack(self.Z)
		else:
			# Perform training integration using IC y0

			# allocate, reshape, normalize, and save solutions
			print('Compute final Y,Z...')
			# now test ability to get YZ solely from integrated r(t) to save solve_ivp computations
			if self.ZY=='new':
				self.newMethod_getYZstuff2(times=t_eval)
				self.newMethod_saveYZ(T_train=T_train)
			else:
				print('Integrating over training data...')
				timer_start = time.time()
				sol_span = solve_ivp(fun=self.rcrf_rhs, t_span=t_span, t_eval=t_span, y0=y0)
				print('...took {:2.2f} minutes'.format((time.time() - timer_start)/60))
				self.newMethod_getYZstuff(yend=sol.y[:,-1])
				self.newMethod_saveYZ(T_train=T_train)

	def newMethod_getYZstuff2(self, times, k=None):

		n_times = times.shape[0]
		dZqq = np.zeros((n_times, self.rf_dim, self.rf_dim))
		dYq = np.zeros((n_times, self.rf_dim, self.output_dim_rf))
		for n in range(n_times):
			t = times[n]
			x = self.x_t(t=t)
			xdot = self.xdot_t(t=t)
			m = self.mscaled(t, x, xdot)
			if self.component_wise:
				x = x[k,None]
				xdot = xdot[k,None]
				m = m[k,None]
			if self.rf_error_input:
				rf_input = np.hstack((x,m))
			else:
				rf_input = x

			q = self.q_t(rf_input)
			dZqq[n] = np.outer(q, q)
			dYq[n] = np.outer(q, m)
		self.Zqq = np.zeros(dZqq.shape[1:])
		self.Yq = np.zeros(dYq.shape[1:])
		for i in range(self.Zqq.shape[0]):
			for j in range(self.Zqq.shape[1]):
				self.Zqq[i,j] = trapz(y=dZqq[:,i,j], x=times)
			for j in range(self.Yq.shape[1]):
				self.Yq[i,j] = trapz(y=dYq[:,i,j], x=times)

	def newMethod_getYZstuff(self, yend):

		if self.component_wise:
			in_dim = 1
		else:
			in_dim = self.input_dim

		Zqq = yend[:self.rf_dim**2]
		Yq = yend[self.rf_dim**2:]
		self.Zqq = Zqq.reshape(self.rf_dim, self.rf_dim)
		self.Yq = Yq.reshape(self.rf_dim, in_dim)


	def newMethod_saveYZ(self, T_train):

		Y = self.Yq
		Z = self.Zqq

		# save Time-Normalized Y,Z
		if self.component_wise:
			self.Y.append(Y / T_train)
			self.Z.append(Z / T_train)
		else:
			self.Y = Y / T_train
			self.Z = Z / T_train

		# store Z size for building regularization identity matrix
		self.reg_dim = Z.shape[0]


	def doNewSolving(self):
		print('Solving inverse problem W = (Z+rI)^-1 Y...')
		# regI = np.identity(self.Z.shape[0])
		regI = np.identity(self.reg_dim)
		regI *= self.regularization_RF

		if self.component_wise:
			# stack regI K times
			regI = np.tile(regI,(self.input_dim,1))

		pinv_ = scipypinv2(self.Z + regI)
		# W_out_all = self.Y.T @ pinv_ # old code
		W_out_all = (pinv_ @ self.Y).T # basically the same...very slight differences due to numerics
		self.W_out_markov = W_out_all

		plotMatrix(self, self.W_out_markov, 'W_out_markov')

		# Compute residuals from inversion
		res = (self.Z + regI) @ W_out_all.T - self.Y
		mse = np.mean(res**2)
		print('Inversion MSE for lambda_RF={lrf} is {mse}'.format(lrf=self.regularization_RF, mse=mse))
		return

	def saveModel(self):
		self.n_trainable_parameters = np.size(self.W_out_markov)
		self.n_model_parameters = np.size(self.W_in_markov) + np.size(self.b_h_markov)
		self.n_model_parameters += self.n_trainable_parameters
		print("Number of trainable parameters: {}".format(self.n_trainable_parameters))
		print("Total number of parameters: {}".format(self.n_model_parameters))

		# print("Recording time...")
		self.total_training_time = time.time() - self.start_time
		print("Total training time is {:2.2f} minutes".format(self.total_training_time/60))

		# print("MEMORY TRACKING IN MB...")
		# process = psutil.Process(os.getpid())
		# memory = process.memory_info().rss/1024/1024
		# self.memory = memory
		# print("Script used {:} MB".format(self.memory))

		data = {
		# "memory":self.memory,
		"n_trainable_parameters":self.n_trainable_parameters,
		"n_model_parameters":self.n_model_parameters,
		"total_training_time":self.total_training_time,
		"W_in_markov":self.W_in_markov,
		"b_h_markov":self.b_h_markov,
		"W_out_markov":self.W_out_markov,
		"scaler":self.scaler,
		"regularization_RF":self.regularization_RF
		# "first_train_vec": self.first_train_vec
		}
		data_path = self.saving_path + self.model_dir + "/data.pickle"
		with open(data_path, "wb") as file:
			pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
			del data
		return 0
