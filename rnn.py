import torch
import torch.nn as nn
import numpy as np
import sys
import os
from utils import *
from typing import List
import pickle as pkl
import pickle

print('...check complete...')

class RNN(nn.Module):

	"""
		Class for initialising an RNN and running the a complete forward pass
	"""

	def __init__(self, params):

		super(RNN, self).__init__()
		self.params = params
		#default is vanilla full-rank RNN
		low_rank = self.params.get("low_rank", False)
		mean_field = self.params.get("mean_field", False)
		use_hook = self.params.get("use_hook", False)

		if low_rank:
			if mean_field:
				if use_hook:
					self.rnn = HookedMeanFieldRNNCell(params)
				else:
					self.rnn = MeanFieldRNNCell(params)
			else:
				if use_hook:
					self.rnn = HookedLowRankRNNCell(params)
				else:
					self.rnn = LowRankRNNCell(params)

		else:
			self.rnn = RNNCell(params)
				
		self.net_size = params["recurrent_size"]
		#set initial network state
		self.init_hidden = nn.Parameter(torch.Tensor(1, params["recurrent_size"]))
		with torch.no_grad():
			#set scale of IC
			try:
				init = params.get("init_scale")
				
				if init:
					self.init_hidden = self.init_hidden.normal_(std=params["init_scale"])
					
				else:
					init_hidden = np.zeros((1, params["recurrent_size"]))
					self.init_hidden = self.init_hidden.copy_(torch.from_numpy(init_hidden))

			except: 

				init_hidden = np.zeros((1, params["recurrent_size"]))
				self.init_hidden = self.init_hidden.copy_(torch.from_numpy(init_hidden))
		

	def forward(self, input_, init_state=None):

		"""
			Performs complete forward pass

		Args:
			input_: the input trials
			init_sate: initial state, defaults to None in which case istance attribute init_hidden is taken as initial state

		Returns:

			rates: network rates
			out: network outputs

		"""
		
		batch_size, trial_len, _ = input_.shape
		outs = torch.zeros(batch_size, 
					 trial_len, 
					 self.params["output_size"], 
					 dtype=torch.float32
					 )
		
		rates = torch.zeros(batch_size, 
					  trial_len, 
					  self.params["recurrent_size"], 
					  dtype=torch.float32
					  )
	
		if init_state is None: 
			hidden_t = torch.tile(self.init_hidden, dims=(batch_size, 1))
		else:
			hidden_t = init_state

		noise = torch.randn(batch_size, 
					  trial_len, 
					  self.params["recurrent_size"], 
					  dtype=torch.float32
					  ) * self.params["noise_amplitude"]

		for idx, input_t in enumerate(input_.split(1, dim=1)): #split batch into single trials
			hidden_t, out = self.rnn(input_t.squeeze(dim=1), hidden_t, noise[:, idx])
			rates[:,idx] = hidden_t
			outs[:,idx] = out

		return rates, outs

class RNNCell(nn.Module):

	def __init__(self, params):

		super(RNNCell, self).__init__()

		self.nonlinearity = set_nonlinearity(params["nonlinearity"])
		self.out_nonlinearity = set_nonlinearity(params["out_nonlinearity"])

		self.w_in = nn.Parameter(torch.Tensor(params["input_size"], params["recurrent_size"]))
		self.w_rec = nn.Parameter(torch.Tensor(params["recurrent_size"], params["recurrent_size"]))
		self.w_out = nn.Parameter(torch.Tensor(params["recurrent_size"], params["output_size"]))
		self.w_in_scale = nn.Parameter(torch.Tensor(1))
		self.w_out_scale = nn.Parameter(torch.Tensor(1, params["output_size"]))
		self.init_hidden = nn.Parameter(torch.Tensor(1, params["recurrent_size"]))
		
		#These will not be trained for now:
		self.w_out_scale.requires_grad = False
		self.w_in_scale.requires_grad = False
		self.init_hidden.requires_grad = False

		if params['train_w_out'] == False:
			self.w_out.requires_grad = False

		self.dt = params['dt']
		self.tau = params['tau']
		self.alpha = params['dt'] / params['tau']
		
		with torch.no_grad():

			self.w_in = self.w_in.normal_()
			#self.w_in = self.w_in.copy_(torch.from_numpy(w_in))

			w_rec = initialize_w_rec(params)
			self.w_rec = self.w_rec.copy_(torch.from_numpy(w_rec))

			#In/Out scaling
			self.w_in_scale = self.w_in_scale.fill_(params["w_in_scale"])
			self.w_out_scale = self.w_out_scale.fill_(params["w_out_scale"])

		#set scaling

			if params["1overN_scale"]:
				self.w_out.normal_(std=1 / params["recurrent_size"])
			else:
				self.w_out.normal_(std=1 / np.sqrt(params["recurrent_size"]))


	def forward(self, input_, hidden_state, noise=0):

		"""
			Performs a single step in the forward pass

		Args:

			input_: the input
			hidden_state: the current state of hidden units

		Returns:

			hidden_state: hidden state after update step
			out: network output
		"""

		#Input to hidden layer:
		rec_input = torch.matmul(
			self.nonlinearity(hidden_state), self.w_rec.t()
		) + self.w_in_scale * input_.matmul(self.w_in)

		noise_t = np.sqrt(2 * self.alpha) * noise
		hidden_state = (1 - self.alpha) * hidden_state + self.alpha * rec_input + noise_t

		#Network output:
		out = self.out_nonlinearity(hidden_state).matmul(self.w_out*self.w_out_scale)

		return hidden_state, out

class LowRankRNNCell(nn.Module):

	"""
	Like RNNCell only the connectivity is constrained to be low-rank
	"""

	def __init__(self, params):

		super(LowRankRNNCell, self).__init__()

		self.params = params
		self.nonlinearity = set_nonlinearity(params["nonlinearity"])
		self.out_nonlinearity = set_nonlinearity(params["out_nonlinearity"])

		self.w_in = nn.Parameter(torch.Tensor(params["input_size"], params["recurrent_size"]))
		#set w_in trainable
		self.w_in.requires_grad = params["train_w_in"]
		#right connectivity vector:
		self.m = nn.Parameter(torch.Tensor(params["recurrent_size"], params["rank"]))
		#set m trainable:
		self.m.requires_grad = True 
		#left connectivity vector:
		self.n = nn.Parameter(torch.Tensor(params["rank"], params["recurrent_size"]))
		#set n trainable 
		self.n.requires_grad = True
		#output vector
		self.w_out = nn.Parameter(torch.Tensor(params["recurrent_size"], params["output_size"]))
		#set w_out trainable
		self.w_out.requires_grad = params["train_w_out"]
		
		#set input/output scales
		self.w_in_scale = nn.Parameter(torch.Tensor(1))
		self.w_out_scale = nn.Parameter(torch.Tensor(1, self.params["output_size"]))
		self.w_out_scale.requires_grad = False
		self.w_in_scale.requires_grad = False
		self.w_inp_scale = self.w_in_scale.fill_(self.params["w_in_scale"])
		self.w_out_scale = self.w_out_scale.fill_(self.params["w_out_scale"])

		#network parameters for convenience
		self.rank = params["rank"]
		self.net_size = params["recurrent_size"]

		self.dt = params["dt"]
		self.tau = params["tau"]
		self.alpha = self.dt / self.tau

		with torch.no_grad():

			if params["LR_method"] is not None:

				try:
					loadings = initialize_lr_connectivity(self.params)
					print("worked")

					m = self.m.copy_(
						torch.from_numpy(
						loadings[params["input_size"] + params["rank"]
						: params["input_size"] + params["rank"] * 2
						].T 
						) 
					)
						
					n = self.n.copy_(
						torch.from_numpy(
						loadings[params["input_size"]: params["input_size"] + params["rank"]]
						) 
					)

					self.w_in = self.w_in.copy_(torch.from_numpy(loadings[:params["input_size"]]))
					self.w_out = self.w_out.copy_(torch.from_numpy(loadings[-params["output_size"]: ]).T)

				#use standard normal
				except Exception as e:
					print(f"The following exception occured: {e}. Reverting to @mastro initialization")

					self.standard_gaussian_parametrization()

			#use standard normal
			else:
				self.standard_gaussian_parametrization()

	def standard_gaussian_parametrization(self):
		#standard initilization following @mastro 2018
		print("used standard gaussian parametrization")
		self.m = self.m.normal_(std=1 / np.sqrt(self.params["recurrent_size"]))
		self.n = self.n.normal_(std=1 / np.sqrt(self.params["recurrent_size"]))
		self.w_in = self.w_in.normal_()
		self.w_out = self.w_out.normal_(self.params["lr_w_out_std"])

		return self

	def forward(self, input_, hidden_state, noise=0):
		#like above but wíth normalised m and n switched in for the recurrent connectivity
		rec_input = torch.matmul(
			torch.matmul(
				self.nonlinearity(hidden_state), self.n.t()), 
				self.m.t()
				) / self.net_size + \
			self.w_in_scale * input_.matmul(self.w_in)
		
		noise_t = np.sqrt(2 * self.alpha) * noise

		hidden_state = (1 - self.alpha) * hidden_state + self.alpha * rec_input + noise_t

		#Network output:
		out = self.w_out_scale * self.out_nonlinearity(hidden_state).matmul(self.w_out) / self.net_size
		return hidden_state, out
	
	@classmethod
	def sample_from_covs(covs, N, params, weights=None):
		"""samples a low-rank RNN from a given set of covariance matrices, one for each population;
		dual of the init_from_covs procedure for MeanFieldRNN;
		Neurons are assigned monotonically. If weights==None, weights are uniform"""

		in_dim = params["input_size"]
		out_dim = params["output_size"]
		num_loadings = covs.shape[1]
		r = int((num_loadings-in_dim-out_dim)/2)
		num_pops = covs.shape[2]

		#loadings
		loadings = np.zeros((num_loadings, N))

		#initialise basis 
		basis_dim = num_loadings * num_pops
		U = np.float32(np.random.randn(basis_dim, N))
		loadings_ls = []
		if weights is None: 
			weights = np.ones(num_pops) * int(N/num_pops)

		for i in range(num_pops):
			low, high = np.cumsum(weights[:i]), np.cumsum(weights[:i+1])
			loadings[:,low:high] = np.float32(np.linalg.cholesky(proj_spd(covs[:,:,i] @ U[:, low:high])))
		
		assert params["low_rank"] == True
		model = RNN(params)

		with torch.no_grad():
			model.rnn.m.set_(torch.from_numpy(loadings[:r,:].T))
			model.rnn.n.set_(torch.from_numpy(loadings[r:2*r,:]))
			model.rnn.w_in.set_(torch.from_numpy(loadings[2*r:2*r+2,:]))
			model.rnn.w_out.set_(torch.from_numpy(loadings[2*r+2:,:].T))
		
		return model
	
	def svd_reparametrization(self):

		"""
		Remove any degeneracy introduced into the recurrent weights during training
		through learned correlations between m and n
		"""
		#ATTENTION: m,n are leaf Variables...
		with torch.no_grad():

			w_rec = self.m.clone() @ self.n.clone()
			w_rec = w_rec.detach().numpy()
			#define non-degenerate m and n as: m = Usqrt(S), n = V.Tsqrt(S)
			U, S, V = np.linalg.svd(w_rec, full_matrices=False)
			#cut off at rank:
			U, S, VT = U[:, :self.rank], S[:self.rank], V[:self.rank,:].T
			self.m.set_(torch.from_numpy(U*np.sqrt(S)))
			self.n.set_(torch.from_numpy(VT*np.sqrt(S)).T)

	def set_w_rec(self):
		"""
		Convenience function for setting recurrent weight matrix from loadings
		"""
		self.__setattr__("w_rec", self.m @ self.n / self.net_size)

class HookedLowRankRNNCell(LowRankRNNCell):

	"""implements LRRNNCell with gradient masking through hooks ="""

	def __init__(self, params):
		super().__init__(params)
		self.num_pops = params.get("num_pops", 1)
		#population weights -> if not provided weights are uniform
		self.weights = params.get("weights", torch.ones(self.num_pops) * (1 / self.num_pops))
		
		#assign neurons monotonically:
		pop_size = int(self.params["recurrent_size"]/self.num_pops)
		self.pop_assignments = torch.arange(self.num_pops).repeat_interleave(pop_size).long()

		#register the population assignments as buffer so it moves into the state dict
		#ATTENTION: might cause problems down the road
		self.register_buffer("pop_assign", self.pop_assignments)

		#set population-level mask
		self.trainable_idxs = params.get("trainable_idxs", None)
		pop_mask = torch.zeros(self.num_pops)
		pop_mask[self.trainable_idxs] = 1.0

		#export to neuron-level mask
		neuron_mask = pop_mask[self.pop_assign] 
		self.register_buffer("neuron_mask", neuron_mask)
		
		#register mask
		self._register_gradient_masks()
		
	def _register_gradient_masks(self):

		"""
		Register the gradient masks -> grads zero-ed out for neurons"""
		if self.m.requires_grad:		
			self.m.register_hook(
            	lambda grad: grad * self.neuron_mask.unsqueeze(1)
        	)
		if self.n.requires_grad:
			self.n.register_hook(
            	lambda grad: grad * self.neuron_mask.unsqueeze(0)
        	)
		if self.w_in.requires_grad:
			self.w_in.register_hook(
            	lambda grad: grad * self.neuron_mask.unsqueeze(0)
        	)
		if self.w_out.requires_grad:
			self.w_out.register_hook(
            	lambda grad: grad * self.neuron_mask.unsqueeze(1)
        	)
	
class MeanFieldRNNCell(nn.Module):

	"""
    Module for simulating and training Mean Fields
	-> trains covariance matrix directly
	"""

	def __init__(self, params):
		super(MeanFieldRNNCell, self).__init__()

		self.params = params
		self.nonlinearity = set_nonlinearity(params["nonlinearity"])
		self.out_nonlinearity = set_nonlinearity(params["out_nonlinearity"])

		self.dt = params["dt"]
		self.tau = params["tau"]
		self.alpha = params["dt"] / params["tau"]
		#number of supports of the network covariance matrix
		self.n_pops = params["n_pops"]
		#loading space dimension for single population
		self.num_loadings = 2 * params["rank"] + params["input_size"] + params["output_size"]
		self.rank = params["rank"]
		self.in_dim = params["input_size"]
		self.out_dim = params["output_size"]
		self.w_out = nn.Parameter(torch.Tensor(params["rank"], params["output_size"]))
		self.weights = nn.Parameter(torch.Tensor(params["n_pops"]))
		self.weights.requires_grad = False
		self.cov_init_noise = params.get("cov_init_noise", 0)

		#set covariance matrices...
		lr_cov = params.get("lr_cov", None)
		covs = np.zeros(
			(self.num_loadings, 
			self.num_loadings, 
			params["n_pops"]
			),
			dtype=np.float32
			)
	
		self.covs = nn.Parameter(torch.Tensor(*covs.shape))

		#...and corresponding cholesky decompositions
		chols = np.zeros_like(covs)
		self.chols = nn.Parameter(torch.Tensor(*covs.shape))

		#MEMO: seems a bit redundant...but could be useful down the road
		with torch.no_grad():
			
			#set weights with balanced pops 
			weights = np.ones(params["n_pops"]) / params["n_pops"]
			self.weights.copy_(torch.from_numpy(weights)) 

			#set w_out with balanced pops
			w_out = np.ones((params["rank"], params["n_pops"])) / params["rank"]*params["n_pops"]
			self.w_out.copy_(torch.from_numpy(w_out))

			if lr_cov is not None:
				
				for i in range(covs.shape[2]):
					cov = lr_cov[...,i]
					params.update({"lr_cov": build_covariance_from_tri(cov)})
					_, chol = initialize_lr_connectivity(params, return_chol=True)
					#Add some noise
					chol += np.random.randn(*chol.shape)*self.cov_init_noise
					chols[:,:,i] = chol
					covs[:,:,i] = chol @ chol.T

			else: 

				for i in range(covs.shape[2]): 
					_, chol = initialize_lr_connectivity(params, return_chol=True)
				
					#Add some noise -> populations different
					chol += np.random.randn(*chol.shape)*self.cov_init_noise
					chols[...,i] = chol
					covs[...,i] = chol @ chol.T

			self.covs.copy_(torch.from_numpy(covs))
			self.chols.copy_(torch.from_numpy(chols))


	def forward(self, input_, hidden_state, noise=0, return_gains=False):

		"""
		Performs forward pass for meanfield equations

		Args:
			input_: torch.Tensor[1 x in_dim]
			hidden_state: torch.Tensor[1 x rank+in_dim]
			noise: noise intensity for additive noise

		Returns:
			hidden_state: the new hidden_state
			out: the corresponding network output

		"""
		
		#start of ns
		nidx = self.rank
		#start of Is
		Iidx = 2*self.rank
		#start of ws
		Widx = 2*self.rank+self.in_dim

		#hidden variables
		K = hidden_state[:, :nidx]
		#input variables
		V = hidden_state[:, nidx:]

		"""
		MEMO:
			l := batch
			k := population
			i,j := regular indices
		"""


		#GAINS:
		gains = self.compute_gains(
				torch.sum(
					torch.hstack(
					(
					torch.einsum('...iik, ...li -> ...lik', self.covs[Iidx:Widx, Iidx:Widx,:], V**2),
					torch.einsum('...iik, ...li -> ...lik', self.covs[:nidx, :nidx, :], K**2)
					)
					), axis=1
				), mode='approx'
			)

		
		#RECURRENT INPUT: 
		rec_input = torch.stack(
    						(torch.einsum('...ijk, ...lj -> ...lik', self.covs[nidx:Iidx, :nidx,:], K),
     						torch.einsum('...ijk, ...lj -> ...lik', self.covs[nidx:Iidx, Iidx:Widx,:], V) 
     						)
						)
		#print(self.covs[nidx:Widx, Iidx:Widx,:])
		#adjust dims for batch
		rec_input *= self.weights * gains.unsqueeze(dim=1)
		K_new = (1 - self.alpha) * K + self.alpha * rec_input.sum(axis=(0,3))
		V_new = (1 - self.alpha) * V + self.alpha * input_ 

		rates = torch.hstack((K_new, V_new))
		
		#want to return output?

		w_out = (torch.einsum('li, ijk -> ljk', K_new, self.covs[:nidx, Widx:,:]) +
		   		torch.einsum('li, ijk -> ljk', V_new, self.covs[Iidx:Widx,Widx:,:]))
		
		
		if self.out_nonlinearity == "tanh":
			w_out = (w_out * self.weights * gains).sum(axis=2)
			
		else:
			w_out = (w_out * self.weights).sum(axis=2)
		
			
		return rates, w_out

	def compute_gains(self, x, mode="approx"):
		"""
		computes gains for populations from latent variables
		uses approximation from Pals et a. (2024) and Marshall (2025) -> 'tanh kinda looks like erf'
		"""
		return 1/ torch.sqrt(1 + (torch.pi / 2) * x)
	
	@classmethod
	def init_from_model(cls, model, pops, cov_init_noise, **kwargs):
		"""
		initialises a mean MeanFieldRNN from a different cell type (right now, only LRRNNCell is supported)
		"""

		params = dict(model.params)
		if pops is None:
			pops = np.zeros(params["recurrent_size"], dtype=int)
		params["mean_field"] = True
		params["n_pops"] = len(set(pops))
		#change to latent variable space
		params["recurrent_size"] = params["rank"] + params["input_size"]

		center = kwargs.get("center", False)
		orth_I = kwargs.get("orth_I", False)
		orth_n = kwargs.get("orth_n", False)

		conn_vecs = get_model_conn_vecs(model, 
								  		center=center, 
										orth_I=orth_I, 
										orth_n=orth_n
										)
		
		covs = get_overlap_matrix_all_params(conn_vecs, pops=pops)
		covs = np.array([proj_spd(covs[...,i]) for i in range(covs.shape[2])]).transpose(1,2,0)
		chols = np.zeros_like(covs)
		
		for i in range(covs.shape[2]):
			params.update({"lr_cov": build_covariance_from_tri(covs[...,i])})
			_, chol = initialize_lr_connectivity(params, return_chol=True)
			
			chol += np.random.randn(*chol.shape)*cov_init_noise
			chols[:,:,i] = chol
			covs[:,:,i] = chol @ chol.T
			
		#clear lr_cov -> bit awkward!
		params.update({"lr_cov": None})
		meanfield_model = RNN(params)
		with torch.no_grad():
			meanfield_model.rnn.chols.copy_(torch.from_numpy(chols))
			meanfield_model.rnn.covs.copy_(torch.from_numpy(covs))
		
		return meanfield_model
	
	@classmethod
	def init_from_covs(cls, params, covs, cov_init_noise=0, **kwargs):

		"""
		Initialises MeanFieldRNN from covariance matrix
		"""

		params = dict(params)
		params["low_rank"] = True 
		params["mean_field"] = True
		#will initialise hooked version if necessary:
		params["use_hooked"] = kwargs.get("use_hooked", False)
		#better here...
		rank = int((covs.shape[1] - params["input_size"] - params["output_size"]) / 2)
		params["rank"] = rank

		#change to latent variable space
		params["recurrent_size"] = rank + params["input_size"]
		
		if len(covs.shape) <= 2:
			covs = np.expand_dims(covs, axis=2)
		
		params["n_pops"] = covs.shape[2]

		#cholesky decomposition for spd-space projected cov matrix
		chols = np.array([np.linalg.cholesky(proj_spd(covs[:,:,i])) for i in range(covs.shape[2])]).transpose(1,2,0)
		chols += np.random.rand(*chols.shape) * cov_init_noise

		"""
		TODO: bit messy -> streamline.
		"""

		#will initialise hooked version if necessary:
		params["use_hooked"] = kwargs.get("use_hooked", False)

		if params["use_hooked"]:
			params["trainable_idxs"] = kwargs.get("trainable_idxs", [])
			params["cov_mask"] = kwargs.get("cov_mask", None)

			#reset trainable populations to random Gaussians or generic matrices
			#define in the params:

			for j in params["trainable_idxs"]: 
					_, chol = initialize_lr_connectivity(params, return_chol=True)
					#Add some noise -> populations different
					chol += np.random.randn(*chol.shape)*cov_init_noise
					chols[...,j] = chol
					covs[...,j] = chol @ chol.T
					
		meanfield = RNN(params)

		with torch.no_grad():
			meanfield.rnn.chols.copy_(torch.from_numpy(chols))
			meanfield.rnn.covs.copy_(torch.from_numpy(covs))

		return meanfield


class HookedMeanFieldRNNCell(MeanFieldRNNCell):

	"""MeanField RNN Cell with masked gradients (implemented through hooks)"""

	def __init__(self, params):

		"""
        Args:
            cov_mask: Entry point for the hook: torch.Tensor[n_loadings, n_loadings, n_supports]
            trainable_k: Entry point for the hook: List := subset([1...n_supports])

		NOTE: Mask will allow training an arbitray set of entries of the [n_loadings x n_loadings x num_pops]
		covariance tensor, so it allows really fine-grained tuning of the training process. 
		Trainable_idxs is more of a quick-and-dirty approach, where anything [...,idx] is trained. 
		Should be enough most of the time. 
        """

		super().__init__(params)
		cov_mask = params.get("cov_mask", None)
		trainable_idxs = params.get("trainable_idxs", None)
	
        # ---- Build mask ----
		if cov_mask is not None:
		
			mask = cov_mask.float()

		elif trainable_idxs is not None:
			mask = torch.zeros_like(self.covs)
			
			#just to be sure:
			idxs = torch.as_tensor(trainable_idxs, dtype=torch.long)
			mask[:, :, idxs] = 1.0 #only x populations trainable

		else:
			#raise ValueError("Either mask or indeces need to be provided")
			#don't raise Error, simply define default behaviour -> will be equivalent MeanFieldRNNCell
			mask = torch.ones_like(self.covs)

		self.mask = mask
		#register the hook
		self.covs.register_hook(self._mask_cov_grad)

	def _mask_cov_grad(self, grad):
		return grad * self.mask
	

class GaussianLRRNN(nn.Module):

	"""Low rank rnn with prefixed population structure. Basic defs like in @Valente,
	where all populations are transformations of a shared Gaussian basis

	For now this is only implemented as a parent class for the hooked version, 
	which allows to train populations separately

	TODO: implement. For now the MF variant serves though and
	a GaussianLRRNN can always be sampled from the basis defined there
	"""

	def __init__(self, params):
		super(GaussianLRRNN, self).__init__()

		pass

#=========CONVENIENCE FUNCTIONS FOR MODEL HANDLING===================

def load_rnn(inPath: str):

	state_dict = os.path.join(inPath, "state_dict.pkl")
	model_params_file = os.path.join(inPath, "model_params.pkl")
	training_params_file = os.path.join(inPath,"training_params.pkl")
	task_params_file = os.path.join(inPath, "task_params.pkl") 

	with open(model_params_file, "rb") as file:
		model_params = pickle.load(file)
	with open(task_params_file, "rb") as file:
		task_params = pickle.load(file)
	with open(training_params_file, "rb") as file:
		training_params = pickle.load(file)

	model = RNN(model_params)

	#manually set path for previously stored models (without path)
	try:
		model_path = model.params["path"]
	except:
		model.params["path"] = inPath

	model.load_state_dict(torch.load(state_dict))

	return model, model_params, task_params, training_params

def save_rnn(outPath: str, 
				model, 
				model_params: dict, 
				task_params: dict, 
				training_params: dict,
				**kwargs):

	model.__setattr__("path", outPath)
	if not os.path.isdir(outPath): 
		os.mkdir(outPath)

	#Safeguard against overwriting
	elif os.path.isdir(outPath):
		outPath += "_2"

	model_params_file = os.path.join(outPath, "model_params.pkl")
	training_params_file = os.path.join(outPath,"training_params.pkl")
	task_params_file = os.path.join(outPath, "task_params.pkl") 

	with open(model_params_file, "wb") as file:
		pkl.dump(model_params, file)

	with open(task_params_file, "wb") as file:
		pkl.dump(task_params, file)

	with open(training_params_file, "wb") as file:
		pkl.dump(training_params, file)

	torch.save(model.state_dict(), os.path.join(outPath, "state_dict.pkl"))

	#save all remaining keyword arguments when recording additional data:
	if kwargs:
		for key, value in kwargs.items():
			file = os.path.join(outPath, f"{key}.npz")
			np.savez(file, value)

def run_rnn(model, input_, init_state=None):

	"""Convenience function for running an RNN of any type
	return detached outputs
	saves on unsqueeze hassle etc. 
	"""

	model.eval()
	if len(input_.shape) <= 2:
		input_ = input_.unsqueeze(dim=0)
	with torch.no_grad(): 
		rates, out = model(input_, init_state=init_state)
	
	rates = rates.detach().numpy()
	out = out.detach().numpy()

	return rates, out

def clone(model):

	"""
	clones a model for experiments
	"""
	model_clone = RNN(model.params)
	model_clone.load_state_dict(model.state_dict())
	return model_clone

def detach_all_parameters(model):

	"""Returns a dictionary with all relevant parameters detached"""

	mparams = {
		name.split('.')[-1]: param.detach().numpy()
		for (name, param) in model.named_parameters()
	}

	return mparams

#=============METHODS MANIPULATING NETWORK STATS========================


def shuffle(model):
	"""
	shuffles a low rank model, thus destroying any population structure
	"""
	params = dict(model.params)
	params["lr_cov"] = get_model_conn_vecs(model, return_labels=False)
	return RNN(params)

def resample(model, N, pops=None, return_pops=False):

	"""
	resample network to dimension N
	ATTENTION: pops are the populations of the to-be-resampled model
	NOT of the resampled model itself
	"""

	num_pops = extract_num_pops(pops)
	params = copy(model.params)
	r = params["rank"]
	#old_size = params["recurrent_size"] -> only for scaling
	params["recurrent_size"] = N

	conn_vecs = get_model_conn_vecs(model, return_labels=False)
	covs = get_overlap_matrix_all_params(conn_vecs, pops=pops)
	num_connectivity_vecs = conn_vecs.shape[0]

	loadings = np.array([np.linalg.cholesky(covs[:,:,i].T) @ np.float32(np.random.randn(num_connectivity_vecs, int(N/num_pops))) for i in range(num_pops)])
	loadings = np.hstack(loadings)

	m = loadings[:r,:] 
	n = loadings[r:2*r,:] 
	w_in = loadings[2*r:2*r+2,:]
	w_out = loadings[2*r+2:,:] 
	model_resampled = RNN(params)

	with torch.no_grad():
		model_resampled.rnn.m.set_(torch.from_numpy(m.T).float()) 
		model_resampled.rnn.n.set_(torch.from_numpy(n).float())
		model_resampled.rnn.w_in.set_(torch.from_numpy(w_in).float())
		model_resampled.rnn.w_out.set_(torch.from_numpy(w_out.T).float())

	if return_pops:
		pops_new = np.repeat(np.array(list(set(pops))), int(N/num_pops))
		return model_resampled, pops_new
		
	else:
		return model_resampled
	
def refactor2pops(model, pops, **kwargs):

	from copy import deepcopy

	"""
	creates a separate model with Gaussian stats for each Gaussian population in a model 
	works just like resample
	"""
	models = []
	r = model.params["rank"]
	orth_I = kwargs.get("orth_I", False)
	orth_n = kwargs.get("orth_n", False)
	conn_vecs = get_model_conn_vecs(model, 
								 return_labels=False,
								 orth_I=orth_I,
								 orth_n=orth_n)

	covs = get_overlap_matrix_all_params(conn_vecs, pops=pops)

	for idx in range(covs.shape[2]):
		params = deepcopy(model.params)
		#just sample with same size
		N = (pops==idx).sum()
		params["recurrent_size"] = N

		loadings = np.array(
					np.linalg.cholesky(covs[:,:,idx].T) 
					@ np.float32(np.random.randn(covs.shape[0], N))
		)

		sub_model = RNN(params)	   
		m = loadings[:r,:] 
		n = loadings[r:2*r,:] 
		w_in = loadings[2*r:2*r+2,:]
		w_out = loadings[2*r+2:,:] 

		with torch.no_grad():
			sub_model.rnn.m.set_(torch.from_numpy(m.T).float()) 
			sub_model.rnn.n.set_(torch.from_numpy(n).float())
			sub_model.rnn.w_in.set_(torch.from_numpy(w_in).float())
			sub_model.rnn.w_out.set_(torch.from_numpy(w_out.T).float())

		models.append(sub_model)

	return models


def pop2network(model, pops, idx):

	"""
	builds a network from a population -> i.e. the network only has the 
	population-specific dynamics. Similar to prev function, only that we simply
	copy the loadings rather than sampling from the distribution

	"""
	params = model.params
	m, n, w_in, w_out = get_lr_net_params(model)
	m_sub = m[:,pops==idx]
	n_sub = n[pops==idx,:]
	w_in_sub = w_in[:,pops==idx]
	w_out_sub = w_out[pops==idx,:]

	#set new network size
	params["recurrent_size"] = (pops==idx).sum()
	sub_model = RNN(params)

	with torch.no_grad():
		sub_model.rnn.m.set_(torch.from_numpy(m_sub.T).float()) 
		sub_model.rnn.n.set_(torch.from_numpy(n_sub).float())
		sub_model.rnn.w_in.set_(torch.from_numpy(w_in_sub).float())
		sub_model.rnn.w_out.set_(torch.from_numpy(w_out_sub.T).float())

	return sub_model

def pops2network(models: List[RNN], model_params) -> RNN:

	"""
	'Fuses' several networks (e.g. populations) into a single network by stacking the
	loadings
	PREREQUISITE: all models have the same rank and the same number of input and
	ouput vectors
	"""

	assert all([model.params["rank"]==models[0].params["rank"] for model in models])
	assert (models[0].net_size==model_params["recurrent_size"])
	assert all([model.rnn.w_in.shape[0]==models[0].rnn.w_in.shape[0] for model in models])
	assert all([model.rnn.w_out.shape[1]==models[0].rnn.w_out.shape[1] for model in models])
	
	m_full = torch.vstack([model.rnn.m for model in models])
	n_full = torch.vstack([model.rnn.n.T for model in models]).T
	w_in_full = torch.vstack([model.rnn.w_in.T for model in models]).T
	w_out_full = torch.vstack([model.rnn.w_out for model in models])

	model_params["recurrent_size"] = sum([model.params["recurrent_size"] for model in models])
	model_fused = RNN(model_params)
	with torch.no_grad():
		model_fused.rnn.m.copy_(m_full)
		model_fused.rnn.n.copy_(n_full)
		model_fused.rnn.w_in.copy_(w_in_full)
		model_fused.rnn.w_out.copy_(w_out_full)

	return model_fused




		


		




	
















