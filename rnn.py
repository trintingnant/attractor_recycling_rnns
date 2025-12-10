import torch
import torch.nn as nn
import numpy as np
import sys
import os
from utils import *
from typing import Optional, Union
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
		#default is vanilla full-rank Rnn
		low_rank = self.params.get("low_rank", False)
		mean_field = self.params.get("mean_field", False)

		if low_rank:
			if mean_field:
				self.rnn = MeanFieldRNNCell(params)
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


		outs = torch.zeros(batch_size, trial_len, self.params["output_size"], dtype=torch.float32)
		rates = torch.zeros(batch_size, trial_len, self.params["recurrent_size"], dtype=torch.float32)
	
		if init_state is None: 
			hidden_t = torch.tile(self.init_hidden, dims=(batch_size, 1))
		else:
			hidden_t = init_state

		noise = torch.randn(batch_size, trial_len, self.params["recurrent_size"]) * self.params["noise_amplitude"]

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
		self.init_hidden.requires_grad = True

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

class MeanFieldRNNCell(nn.Module):

	"""
    Module for simulating and training Mean Fields
	trains Covariance matrix directly
	"""

	def __init__(self, params):
		super(MeanFieldRNNCell, self).__init__()
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

		#set covariance matrices
		covs = np.zeros(
			(self.num_loadings, 
			self.num_loadings, 
			params["n_pops"]
			),
			dtype=np.float32
			)
		
		#and corresponding cholesky decompositions
		self.chols = nn.Parameter(torch.Tensor(*covs.shape))

		with torch.no_grad():
			
			#set weights with balanced pops 
			weights = np.ones(params["n_pops"]) / params["n_pops"]
			self.weights.copy_(torch.from_numpy(weights)) 

			#set w_out with balanced pops
			w_out = np.ones((params["rank"], params["n_pops"])) / params["rank"]*params["n_pops"]
			self.w_out.copy_(torch.from_numpy(w_out))

			#bit awkward. Requires post-hoc setting. Let's see. 
			self.chols.copy_(torch.from_numpy(covs))

	def forward(self, input_, hidden_state, noise=0):

		"""
		Performs forward pass for meanfield equations

		Args:
			input_: torch.Tensor[batch_size x in_dim]
			hidden_state: torch.Tensor[1 x rank+in_dim]
			noise: noise intensity for additive noise

		Returns:
			hidden_state: the new hidden_state
			out: the corresponding network output

		"""
		#hidden variables
		K = hidden_state[:, :self.rank]
		#input variables
		V = hidden_state[:, self.rank:]
		
		#start of ns
		nidx = self.rank
		#start of Is
		Iidx = 2*self.rank
		#start of ws
		Widx = 2*self.rank+self.in_dim

		#GAINS:
		#recurrent
		gains = torch.einsum('iik, ji -> jik', self.chols[:nidx, nidx:Iidx, :], K**2)
		#input
		gains += torch.einsum('iik, ji -> jik', self.chols[Iidx:Iidx+Widx, Iidx:Widx,:], V**2)
		gains = self.compute_gains(gains, mode='approx')
		"""
		gains = self.compute_gains(
			torch.einsum('iik, ik -> k', self.chols[:nidx, nidx:Iidx, :], K**2) +
			torch.einsum('iik, ik -> k' self.chols[Iidx:Iidx+Widx, Iidx:Widx,:], V**2),
			mode='approx'
		)

		rec_input = torch.sum(
			self.weights *
			(torch.einsum('ijk, jk -> ik', self.chols[nidx:Iidx, :nidx,:], K) +
			torch.einsum('ijk, jk -> ik', self.chols[Iidx:Widx, Iidx:Widx,:], V)
			),
			axis=1
		)

		"""

		#recurrent input
		rec_input = torch.einsum('ijk, j -> ik', self.chols[nidx:Iidx, :nidx,:], K)
		#input
		rec_input += torch.einsum('ijk, j -> ik', self.chols[Iidx:Widx, Iidx:Widx,:], V)
		rec_input *= self.weights

		K_new = (1 - self.alpha) * K + self.alpha * (gains * rec_input).sum(axis=1)
		V_new = (1 - self.alpha) * V + self.alpha * input_ 

		#want to return output?
		"""
		w_out = torch.einsum('ijk, jk -> ik', self.chols[:nidx, Widx:,:], K)
		w_out += torch.einsum('ijk, jk -> ik', self.chols[Iidx:Widx, Widx,:], V)
		out = w_out * gains * self.weights
		"""

		return torch.cat((K_new, V_new), 1)

	def compute_gains(self, x):
		"""
		computes gains for populations from latent variables
		"""
		return 1/ torch.sqrt(1 + torch.pi / 2 * x)
	
	@classmethod
	def init_from_model(cls, model, pops):
		"""
		initialises a mean MeanFieldRNN from a different cell type (right now, only LRRNNCell is supported)
		"""

		params = dict(model.params)
		params["mean_field"] = True
		params["n_pops"] = len(set(pops))

		conn_vecs = get_model_conn_vecs(model, center=True)
		covs = get_overlap_matrix_all_params(conn_vecs, pops=pops)

		#Cholesky decomposition: use L.T instead of L
		mf_model = RNN(params)
		with torch.no_grad():
			mf_model.rnn.chols.copy_(torch.from_numpy(covs.transpose(1,0,2)))
							
		return mf_model


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
	if os.path.isdir(outPath):
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

def clone(model):

	"""
	clones a model for experiments
	"""
	model_clone = RNN(model.params)
	model_clone.load_state_dict(model.state_dict())
	return model_clone

def detach_all_parameters(model):

	"""Returns a copy of the model with all relevant parameters detached"""
	pass

def shuffle(model):
	"""
	shuffles a low rank model, thus destroying any population structure
	"""
	params = dict(model.params)
	params["lr_cov"] = get_model_conn_vecs(model, return_labels=False)
	return RNN(params)

def resample(model, N, pops=None):

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

    return model_resampled

def compute_gains(x, mode='approx'):
	"""
	Computes the gains for LR_RNN models, either (for LRRNN_Cells) by sampling, or by using approximation from @Pals et al. 2023
	
	Args:

		x: torch.Tensor 
		mode: either "approx" or "sample"

	Returns:
		gain-modulated activities of neurons ("sample") or gain-modulated latent variables ("approx")

	TODO: implement "sample" functionality
	"""

	if mode=='approx':
		return 1 / torch.sqrt(1 + torch.pi/2 * x)









