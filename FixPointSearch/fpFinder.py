"""Module for finding RNN fix points"""
import torch, numpy as np
from rnn import RNN as RNN
from torch.autograd import Variable, functional
import os, pickle as pkl
from FixedPoint import FixedPoint, TensorArray


class FixedPointFinder(object):

	def __init__(self, 
				model: RNN,
				fpf_params: dict,
				lrs_params: dict,
				lr: bool=False, 
				verbose: bool=True,
				super_verbose: bool=False
				):
		self.model = model
		self.gamma = fpf_params['gamma']
		self.speed_tor = fpf_params['speed_tor']
		self.max_epochs = fpf_params['max_epochs']
		self.trnst = fpf_params['trnst']
		self.lr = lr
		self.lr_decay_epoch = fpf_params['lr_decay_epoch']
		self.d_speed_tor = fpf_params['d_speed_tor']
		self.random_seed = fpf_params['random_seed']
		self.rng = np.random.RandomState(self.random_seed)
		self.lr_decay_schedule = lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(
			optim,
			mode=lrs_params['mode'],
			factor=fpf_params['lr_decay_factor'],
			patience=lrs_params['patience'],
			cooldown=lrs_params['cooldown']
			)
		self.noise_scale = fpf_params['noise_scale']

		#Set model into evaluation mode
		self.model.eval()



	def calc_speed(self, hidden, input_):

		"""
		Calculates speed/energy of the dynamics. 

		Args:

			hidden: torch.Tensor[net_size x net_size] -> the hidden state
			input_: torch.Tensor[in_dim x trial_len] -> should be a constant signal

		Returns:

			speed: the speed/energy @hidden @input_

		"""

		#TODO: adjust for LowRankNets, right now it works because you can always set w_rec and use as for FullRankNets

		if self.lr:
			rec_input = torch.matmul(
			torch.matmul(
				self.model.rnn.nonlinearity(hidden), self.model.rnn.n.t()), 
				self.model.rnn.m.t()
				) / self.model.rnn.net_size + \
			self.model.rnn.w_in_scale * input_.matmul(self.model.rnn.w_in)

		else:
			rec_input = torch.matmul(
				self.model.rnn.nonlinearity(hidden), self.model.rnn.w_rec.t()
				) + self.model.rnn.w_in_scale * input_.matmul(self.model.rnn.w_in)

		activation = (1 - self.model.rnn.alpha) * hidden + self.model.rnn.alpha * rec_input
		speed = torch.norm(activation - hidden)

		return speed

	def find_fixed_point(self,
						input_,
						max_iterations,
						init_hidden=None,
						record_hiddens=False,
						verbose=False,
						super_verbose=False
						):

		"""
		Finds FP/FPS of the dynamics starting from stats randomly sampled from the trajectory

		Args:

			init_hidden: torch.Tensor[net_size] -> the initial condition, if 'None' ICs will be a sample of the trajectory + noise
			input_: torch.Tensor[in_dim x trial_len] -> should be constant signal
			max_iterations: int -> maximum number of resampling runs

		Returns:

			FP: the fixed point
			check: did the optimisation terminate?
			hiddenS: the optimisation trajectory in the neural activity space

		TODO:

			1. Add noisy sampling [done in calling function]
			2. Add adaptive speed_tolerance (otherwise very small gradient steps resulting from the adaptive learning rate may yield spurious results)

		"""

		FP = FixedPoint._setup(self.model, input_)
		iteration_count = 0
		speed_prev = torch.tensor([0])
		FP.x_init = init_hidden if init_hidden is not None else self.sample_from_rates(FP.rates)
		hidden_ = FP.x_init.clone()
		hiddens = [hidden_]

		while iteration_count <= max_iterations:
			print('start')

			counter = 0
			iteration_count += 1

			if FP._check == True:
				FP.x_star = hidden_
				FP.qstar = speed_prev
				try:
					FP.dqstar = d_speed
				except:
					pass

				return FP

			else:
				optimizer = torch.optim.Adam([FP.x_init], lr=self.gamma)
				scheduler = self.lr_decay_schedule(optimizer)

				while True:
					hidden_activation = Variable(hidden_)
					hidden_activation.requires_grad = True
			
					speed = self.calc_speed(hidden_activation, input_)
					d_speed = speed_prev - speed
					speed_prev = speed.clone()

					#need to retain graph here
					speed.backward(retain_graph=True)
					optimizer.step()
					scheduler.step(metrics=speed)

					if counter % self.lr_decay_epoch == 0 and counter != 0:
						if verbose:
							print(f'epoch: {counter}, speed={speed.item()}')

					if speed.item() < self.speed_tor[1]:
						print('speed')
						print(f'epoch: {counter}, speed={speed.item()}')
						FP._check = True
						break

					if np.abs(d_speed.item()) < self.d_speed_tor and counter >= 1:
						
						#Still accept if the point is reasonably slow
						if speed.item() < self.speed_tor[0]:
							print(f'epoch: {counter}, acceleration={d_speed.item()}')
							print(f'epoch: {counter}, speed={speed.item()}')
							FP._check = True
							break

						elif iteration_count < max_iterations:
							print(f'speed no longer improving after {counter} steps. Final speed: {speed}.\nRestarting fix point search from resampled IC: {iteration_count}')
							#resample from trajector
							hidden_ = self.sample_from_rates(FP.rates)
							FP.x_init = hidden_
							break

						else: 
							break


					if counter == self.max_epochs:
						print(f'Iteration count exceeds {self.max_epochs}. Forcibly terminating fix-point search.')
						print(f'Restarting fix point search from resampled IC: {iteration_count}')
						#still accept if reasonably slow
						if speed.item() < self.speed_tor[0]:
							print(f'epoch: {counter}, speed={speed.item()}')
							FP._check = True
							break
 
						FP._check = False
						break

					counter += 1
					lr = optimizer.param_groups[0]['lr']
					hidden_ = hidden_activation - lr * hidden_activation.grad

					if record_hiddens:
						FP.hiddens.append(hidden_)

		raise FixedPointSearchFailureException('Fix point search failed')

	def sample_from_rates(self, rates):
		#TODO: should be parametrised by 
		"""
		Samples a single initial condition from a given trajectory
		and perturbs with noise
		"""
		noise_scale = self.noise_scale
		#TODO: implement for multi-batch FP search
		_, trial_len, net_size = rates.shape
		rates = rates.detach().clone()
		init_hidden = rates[:, self.rng.choice(range(self.trnst, trial_len)), :] + \
					torch.randn(*rates[:,0,:].shape) * noise_scale
		#kind of hacky leaf creation -> TODO: find better way
		init_hidden = init_hidden.detach()
		init_hidden.requires_grad = True
		return init_hidden

	def fixed_point_grid_search_lr(self, input, num_grid_points, dist_thresh):
		"""
		calculates suitable ICs for FP grid search for low rank nets
		"""
		from utils import proj_
		from math import pow
		r = self.model.params["rank"]
		noise_scale = self.noise_scale
		rates, _ = self.model(input.unsqueeze(dim=0))

		"""
		#scaling factors between Euclidean/Overlap space
		scale_facs = [model.rnn.net_size / m[:,i] @ m[:,i] for i in range(r)]
		"""

		#distance function for weeding out duplicates
		dist = lambda x, y: torch.linalg.norm(x-y)
		m = self.model.rnn.m.clone().detach().numpy()
		ks = proj_(m.T, rates.detach().numpy())
		N = int(pow(num_grid_points, 1/r))
		grid = [np.linspace(ks[i,:].min(), ks[i,:].max(), N) for i in range(r)]
		points_grid = np.meshgrid(*grid)

		I = (input @ self.model.rnn.w_in.detach().numpy())[0,:]
		#Gram-Schmidt-Orthogonalisation
		for i in range(r):
			I = I - (I @ m[:,i]) * m[:,i] / (m[:,i] @ m[:,i])
		I_orth = I.detach().numpy()
		fps = []
		for j in range(points_grid[0].size):
			#Determine points in the overlap space
			gp = tuple(points_grid[i].ravel()[j] for i in range(r))
			#initial condition
			ic = np.array([gp[i] * m[:,i] for i in range(r)]).sum(axis=0) + I_orth
			try:
				FP = self.find_fixed_point(input,
									max_iterations=3,
									init_hidden=torch.from_numpy(ic),
									record_hiddens=True,
									verbose=True,
									super_verbose=True
									)

				#weed out duplicates:
				if not (np.array([dist(FP.x_star, fp.x_star) < dist_thresh for fp in fps])).any():
					fps.append(FP)

			except:
				continue



		return fps


	def fixed_point_grid_multi_input_sweep_lr(self, inputs, num_grid_points, dist_thresh):

		fps = [] 
		for idx in range(inputs.shape[0]):
			FPS = self.fixed_point_grid_search_lr(inputs[idx,:,:], num_grid_points, dist_thresh)
			fps.append(FPS)
			#should this be ravelled? -> makes it harder to determine identity of input post

		return fps

	def seq_fixed_point_search(self,
								inputs,
								max_iterations,
								init_hidden=None,
								verbose=True,
								super_verbose=False
								):

		"""Sequential fixed points search"""

		n_inputs = inputs.shape[0] if type(inputs)==TensorArray else len(inputs)
		fps = []
		for input_idx in range(n_inputs):
			try:
				print(f'Starting fix point seach for input: {input_idx}')
				FP = self.find_fixed_point(inputs[input_idx, :, :],
											max_iterations=max_iterations,
											init_hidden=init_hidden,
											verbose=verbose,
											super_verbose=super_verbose
											)

				FP.calc_jacobians_xstar()
				fps.append(FP)

			except FixedPointSearchFailureException as e:
				print(f'Attention: fix point search failed after {max_iterations} iterations')
				fp = FixedPoint._setup(self.model, inputs[input_idx, ...])
				fps.append(fp)

		return fps


class FixedPointSearchFailureException(Exception):

	def __init__(self, message):
		self.message = message
		super().__init__(self.message)

	def __str__(self):
		return f'{self.message}'
