import numpy as np, torch.nn as nn
import torch
import os, pickle as pkl
from typing import Optional
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from copy import copy



"""
Bag of RNN-related convenience functions
"""

#==========INITIALISERS=============:

def initialize_w_rec(params):

	w_rec = np.zeros((params["recurrent_size"], params["recurrent_size"]))

	w_rec = np.random.normal(0,1, (params["recurrent_size"], params["recurrent_size"])) \
		* params["spectral_radius"] \
		/ np.sqrt(params["recurrent_size"])
	
	#TODO: add connectivity mask -> just for testing behaviour under
	#different network topologies
	"""
	if params.get('mask') and params['mask']:
		mask = get_conn_mask(*w_rec.shape)
		w_rec = w_rec * mask
	"""

	#TODO: Use other distributions -> only possible if:
	#i) the distribution has negative support 
	#ii) we have inhibitory neurons

	#Set to spectral radius:

	if params["spectral_norm"]:
		w_rec = (
			params["spectral_radius"]
			* w_rec
			/ np.max(np.abs((np.linalg.eigvals(w_rec))))
		)

	return w_rec


def initialize_w_in(params):

	return np.random.normal((params["input_size"], params["recurrent_size"]))


def initialize_lr_connectivity(params, return_chol=False):

	"""
	Initiliazes ladings for low_rank RNNs

	"""
	if params["LR_method"] == "mastro":

		#default: standard normal left and right eigenvectors
		define_random_connectivity = lambda x: x.normal_() / np.sqrt(params["recurrent_size"])
		return define_random_connectivity(m), define_random_connectivity(n)

	elif params["LR_method"] == "cholesky":

		#Uses cholesky decomposition to return correlated left and right eigenvectors
		#i.e. performs ch. decomp. of the covariance matrix of all connectivity vectors
		num_connectivity_vecs = params["rank"] * 2 + params["input_size"] + params["output_size"]
		if params["lr_cov"] is None:
   			# generate random covariance matrix
   			# with n and m correlated to avoid vanishing gradients
			cov = np.eye(num_connectivity_vecs)	
			for i in range(params["rank"]):

      			#Set the individual factors
				cov[params["input_size"] + i, params["input_size"] + params["rank"] + i] = params['low_rank_cov']
      
				cov[params["input_size"] + params["rank"] + i, params["input_size"] + i] = params['low_rank_cov']

		else:
			#if covariance matrix for cholesky decomposition is provided
			cov = params["lr_cov"]
		
		lower_triangular = np.float32(np.linalg.cholesky(cov))
		#compute the loadings by projecting up into R^n
		loadings = lower_triangular @ np.float32(np.random.randn(num_connectivity_vecs, params["recurrent_size"]))

		if return_chol: 
			return loadings, lower_triangular
		else: 
			return loadings

	elif params["LR_method"] == "frschu":

		#Method for initialising connectivity matrix where lr-components 
		#are correlated with the random part of the connectivity
		#TODO: not really useful right now, since I'm using J = N(0, g^2) + P ~ P for small g. 
		#So I don't have a random background
		raise NotImplemented


def set_nonlinearity(nonlinearity: str):


	if nonlinearity == "tanh":
		return torch.tanh

	if nonlinearity == "sigmoid":
		return torch.sigmoid

	elif nonlinearity == "identity":
		return lambda x: x

	else:
		print('Unknown non-linearity. Continuing with identity')
		return lambda x: x

def mse(out, target, mask):
	
	loss = torch.sum((target[mask] - out[mask])**2) / mask.sum()
	return loss

def zero_loss():

	return torch.zeros(1)


#=============MISCELLANEOUS====================

def reset_params(name, value, file):
	"""
	Convenience function for resetting some params for importing sync:
	dumps new params to old location
	"""
	with open(file, "rb") as file: 
		params = pkl.load(file)

	params[name] = value

	with open(file, "wb") as file:
		pkl.dump(params, file)

def set_wrec(model):
	model.rnn.__setattr__("w_rec", model.rnn.m @ model.rnn.n / model.rnn.net_size)


def get_unstable_subspace(model, orth=True):

	import warnings

	w_rec = model.rnn.w_rec.detach().numpy()
	eigvecs, eigvals = np.linalg.eig(w_rec)[::-1]
	real, imag = np.real(eigvals), np.imag(eigvals)
	#get all unstable directions
	idxs = np.where(real>1)[0]
	
	
	i = 0
	unstable_	 = []
	while i < idxs.shape[0]:
		if imag[idxs[i]] != 0:
			rconj = np.real(eigvecs[:, idxs[i]])
			iconj = np.imag(eigvecs[:, idxs[i]])
			unstable_dirs.extend([rconj, iconj])
			i += 2

		else:
			unstable_dirs.append(eigvecs[:,idxs[i]])
			i += 1

	unstable_dirs = np.array(unstable_dirs)
	if orth:
		unstable_dirs = gram_schmidt_vecs(unstable_dirs)

	#cast as real, imaginary part is 0 anyway  -> ignore warning
	with warnings.catch_warnings(action="ignore"):
		return np.array(unstable_dirs, dtype=np.float64)


def gram_schmidt_matr(vecs):

	"""
	Returns an orthonormal basis for a set of vectors passed as a matrix.
	Vectors for orthogonalisation appear as row vectors
	Attention: operations are in-place
	"""

	vecs[0,:] = vecs[0,:] / np.linalg.norm(vecs[0,:])

	for idx in range(1, vecs.shape[0]):
		#don't need to normalise here, as current basis is already orthonormal up to idx
		vecs[idx,:] = vecs[:idx,:].T @ vecs[:idx,:] @ vecs[idx,:]
		#normalise the final thing
		vecs[idx,:] /= np.linalg.norm(vecs[idx,:])


def gram_schmidt_vecs(vecs):

    vecs_orth = []
    vecs_orth.append(vecs[0] / np.linalg.norm(vecs[0]))

    for i in range(1, len(vecs)):

        v = vecs[i]
        for j in range(i):
            v = v - (v @ vecs_orth[j]) * vecs_orth[j]

        v = v / np.linalg.norm(v)
        vecs_orth.append(v)

    return vecs_orth

def orth_proj(x,y):
	"""
	computes orthogonal projection of y on x
	"""
	vecs = np.vstack((x,y))	
	return gram_schmidt_vecs(vecs)[:,1]

def get_orth_proj_mat(X): 
	"""
	computes orthogonal projector matrix for a given space X
	"""
	return X @ np.linalg.inv(X.T @ X) @ X.T

def proj_(ms, x):
	""" Projection onto connectivity vectors for low-rank RNNs"""
	
	k = np.zeros((ms.shape[0], x.shape[1]))
	for mIdx, m in enumerate(ms):
		k[mIdx,:] = x[0,:,:] @ m
		k[mIdx,:] /= np.linalg.norm(m)**2
	return k

def phiPrime(x):
	"""derivative of tanh activation function"""
	return 1 - np.tanh(x)**2

def get_lr_net_params(model):

	m = model.rnn.m.clone().detach().numpy()
	n = model.rnn.n.clone().detach().numpy()
	w_in = model.rnn.w_in.clone().detach().numpy()
	w_out = model.rnn.w_out.clone().detach().numpy()

	return m, n, w_in, w_out

def get_overlap_matrix(model):
	"""obtain overlap matrix for low-rank RNNs"""

	r = model.rnn.rank
	N = model.rnn.net_size
	m, n, _, _ = get_lr_net_params(model)
	sov = np.zeros((r, r))
	for i in range(r):
		for j in range(r):
			sov[i,j] = m[:,i] @ n[j,:] / N
	return sov

def get_input_correlation_matrix(model):
	"""obtain the input correlation matrix for low-rank RNNs, i.e.
	the matrix of overlaps between the input patterns
	and the input-selection vectors n"""

	r = model.rnn.rank
	N = model.rnn.net_size
	in_dim = model.params["input_size"]
	_, n, w_in, _ = get_lr_net_params(model)
	sni = np.zeros((in_dim, r))
	for j in range(r):
		sni[:,j] = w_in.dot(n[j,:]) / N

	return sni

def get_model_conn_vecs(model, 
						orth_I=False, 
						center=False,
						orth_n=False, 
						return_labels=False):
	
	from copy import deepcopy

	"""returns the the 2R + N_in + N_out connectivity params for a low rank model
	stack in preparation for GMM fit
	"""

	conn_vecs = np.zeros((2*model.rnn.rank
                      + model.rnn.params["input_size"]
                      + model.rnn.params["output_size"], 
                      model.rnn.net_size))

	m, n, w_in, w_out = get_lr_net_params(model)
	
	#keep only the orthogonal part of I -> no drive along m
	#Ref: Valente 2019, Dubreuil Valente et al. 2022, etc.
	if orth_I:
		#get input directions
		for i in range(w_in.shape[0]):
			vecs = np.vstack((m.T, w_in[i,:]))
		I_orth = gram_schmidt_vecs(vecs)
		w_in = I_orth[1:,:]
	
	#Only keep parallel part of n:
	if orth_n:
		orth_n_vec = deepcopy(n)
		for idx, nI in enumerate(orth_n_vec):
			vecs = np.vstack((m.T, w_in, nI))
			nI_orth = gram_schmidt_vecs(vecs)[-1,:]
			orth_n_vec[idx, :] = nI_orth

	conn_vecs[:model.rnn.rank,:] = m.T
	conn_vecs[model.rnn.rank: 2*model.rnn.rank,:] = n
	conn_vecs[2*model.rnn.rank:2*model.rnn.rank + model.rnn.params["input_size"],:] = w_in
	conn_vecs[2*model.rnn.rank + model.rnn.params["input_size"]:,:] = w_out.T 

	if center:
		conn_vecs -= conn_vecs.mean(axis=1, keepdims=True)

	if return_labels:
		return conn_vecs, build_labels(model)
	else: 

		return conn_vecs

def get_overlap_matrix_all_params(conn_vecs, pops=None):

	"""obtains overlap matrix for all connectivity parameters, 
	including inputs and outputs
	"""

	num_params = conn_vecs.shape[0]

	if pops is None:
		pops = np.zeros(conn_vecs.shape[1], dtype=int)

	ov_mat = np.zeros((num_params, num_params, len(set(pops))))

	for elem in set(pops):
		print(elem)
		_, counts = np.unique(pops, return_counts=True)
		mask = np.where(pops==elem)[0]
		for i in range(num_params):
			for j in range(i, num_params):
				ov_mat[i,j, elem]= (conn_vecs[i,mask] * conn_vecs[j, mask]).sum() / counts[elem]

	return ov_mat

def build_covariance_from_tri(cov):
    return cov + cov.T - np.eye(cov.shape[0])*np.diag(cov)

def get_eff_overlap_matrix(model, fp):
	"""
	obtains effective recurrent weight matrix under input modulation
	"""

	from utils import phiPrime

	r = model.rnn.rank
	m, n, w_in, _ = get_lr_net_params(model)

	if fp._check:
		x = fp.x_star.detach().numpy()
	else:
		print("Overlap matrix could not be computed. Likely cause: missing fixed point.")
		return None

	effN = np.array([n[i,:] * phiPrime(x) for i in range(r)])
	ov_jac = effN[:,0,:] @ m / model.rnn.net_size
	try:
		res = (np.array([np.outer(m[:,i], effN[i,0,:])for i in range(r)])).sum() \
						/ model.rnn.net_size - np.eye(r)

	except:
		print("Overlap matrix could not be computed. ")
		return None

	return res, ov_jac

def extract_num_pops(pops):
	return len(set(pops))

def build_labels(model):

	"""builds connectivity vector labels for plotting etc."""

	return ([f"m$_{i}$" for i in range(1,model.rnn.rank+1)] 
			+ [f"n$_{i}$" for i in range(1,model.rnn.rank+1)]
			+ [f"I$_{i}$" for i in range(1,model.rnn.params["input_size"]+1)]
			+ [f"w$_{i}$" for i in range(1,model.rnn.params["output_size"]+1)]
			)

def build_model_dataframe(model, pops):

	"""builds model dataframe for sns interfacing"""

	import pandas as pd
	
	conn_vecs, labels = get_model_conn_vecs(model, return_labels=True)
	df = pd.DataFrame(conn_vecs.T, columns=labels)
	df['pops'] = pops
	
	return df

















