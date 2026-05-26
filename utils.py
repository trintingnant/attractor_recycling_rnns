import numpy as np, torch.nn as nn
import torch
import os, pickle as pkl
from typing import Optional, Iterable
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from copy import copy, deepcopy


"""
Bag of RNN-related utils/convenience functions
"""

#==========INITIALISERS=============:

def initialize_w_rec(params):

	w_rec = np.zeros((params["recurrent_size"], params["recurrent_size"]))

	w_rec = np.random.normal(0,1, (params["recurrent_size"], params["recurrent_size"])) \
		* params["spectral_radius"] \
		/ np.sqrt(params["recurrent_size"])
	
	#TODO: add connectivity mask -> just for testing behaviour under
	#different network topologies. Ref: Herbert & Ostojic 2022
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


"""Define Regularizers"""

def l2_loss(rates, **kwargs):
	"""L2 loss on single unit firing rates"""
	return torch.mean(rates**2)

def l1_loss(rates, **kwargs):
	"""L1 loss on single unit firing rates"""
	return torch.mean(rates.abs())

def l2_cov_loss(rates, **kwargs):
	"""Promotes covariance matrices with small entries"""
	covs = kwargs.get("covs", None)
	l2_loss = torch.mean(covs**2)
	return l2_loss.squeeze()

def l1_cov_loss(rates, **kwargs):
	"""Promotes covariance matrices with lots of 0s"""
	covs = kwargs.get("covs", None)
	l1_loss = torch.mean(covs.abs())
	return l1_loss.squeeze()

def build_regularisers(regs):

	"""
	builds list of regulariser functions to pass to training function
	"""
	fns = [l1_loss, l2_loss, l1_cov_loss, l2_cov_loss]
	reg_fns = [rf for (r, rf) in zip(regs, fns) if r]
	return reg_fns
	

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

def compute_eff_rank(model, eps=1e-3):

    """Computes the effective rank of the model recurrent weight matrix, 
    defined as exponentiated (Shannon) entropy of the (normalised) singular values"""
    
    if not hasattr(model.rnn, "w_rec"):
        #ATTENTION: will set attribute globally
        model.rnn.set_w_rec()
    w_rec = model.rnn.w_rec.detach().numpy()
    sigmas = np.linalg.svd(w_rec, compute_uv=False)
    norm = sigmas.sum()
    if norm > eps: 
        sigmas /= norm
        sigmas = sigmas[sigmas > eps]
    else:
        raise ValueError("Could not be computed. Norm of singular values too small")

    #filter out all 0 singular values -> import for LR
    sigmas = sigmas[sigmas > eps]
    H = -(sigmas * np.log(sigmas)).sum()
    
    return np.exp(H)

def detach_tensors(txs: Iterable[torch.Tensor]) -> Iterable[np.ndarray]:
	return (x.detach().numpy() for x in txs)

def get_unstable_subspace(model, orth=True):

	import warnings

	w_rec = model.rnn.w_rec.detach().numpy()
	eigvecs, eigvals = np.linalg.eig(w_rec)[::-1]
	real, imag = np.real(eigvals), np.imag(eigvals)
	#get all unstable directions
	idxs = np.where(real>1)[0]
	
	i = 0
	unstable_dirs	 = []
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

	#orthogonalise
	if orth:
		unstable_dirs = gram_schmidt_vecs(unstable_dirs)

	#cast as real, imaginary part is 0 anyway  -> ignore warning
	with warnings.catch_warnings(action="ignore"):
		return np.array(unstable_dirs, dtype=np.float64)


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
	
	k = np.zeros((x.shape[0], ms.shape[0], x.shape[1]))

	for i in range(x.shape[0]):
		for mIdx, m in enumerate(ms):
			k[i,mIdx,:] = x[i,:,:] @ m
			k[i,mIdx,:] /= np.linalg.norm(m)**2
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

def get_model_conn_vecs(model, 
						orth_I=False, 
						center=False,
						orth_n=False, 
						return_labels=False):
	
	from copy import copy, deepcopy

	"""returns the the 2R + N_in + N_out connectivity params for a low rank model
	stack in preparation for GMM fit + general connectivity space analysis
	"""

	conn_vecs = np.zeros((2*model.rnn.rank
                      + model.rnn.params["input_size"]
                      + model.rnn.params["output_size"], 
                      model.rnn.net_size),
					  dtype='float32')

	m, n, w_in, w_out = get_lr_net_params(model)
	
	#keep only the orthogonal part of I -> no drive along m
	#Ref: Valente 2019, Dubreuil Valente et al. 2022, etc.
	w = np.zeros_like(w_in)	
	if orth_I:
		for idx, inI in enumerate(w_in):
			vecs = np.vstack((m.T, inI))
			w[idx,:] = np.array(gram_schmidt_vecs(vecs))[-1,:]

	else:
		w = copy(w_in)

	if orth_n:
		nn = np.zeros_like(n)
		for idx, nI in enumerate(n):
			vecs = np.vstack((m.T, w, nI))
			nn[idx, :] = np.array(gram_schmidt_vecs(vecs))[-1,:]

		n -= nn 


	conn_vecs[:model.rnn.rank,:] = m.T
	conn_vecs[model.rnn.rank: 2*model.rnn.rank,:] = n
	conn_vecs[2*model.rnn.rank:2*model.rnn.rank + model.rnn.params["input_size"],:] = w
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
	ATTENTION: this will yield an upper triangular matrix for plotting interfacing
	to obtain full covariances run the function below
	TODO: should be integrated.
	"""

	num_params = conn_vecs.shape[0]

	if pops is None:
		pops = np.zeros(conn_vecs.shape[1], dtype=int)

	ov_mat = np.zeros((num_params, num_params, len(set(pops))))

	for elem in set(pops):
		_, counts = np.unique(pops, return_counts=True)
		mask = np.where(pops==elem)[0]
		for i in range(num_params):
			for j in range(num_params):
				cov = np.sum(
					(conn_vecs[i, mask] - conn_vecs[i, mask].mean()) *
		   			(conn_vecs[j, mask] - conn_vecs[j, mask].mean())
				)
						   
				ov_mat[i,j, elem] = cov / counts[elem]

	return ov_mat

def build_covariance_from_tri(cov):
    return cov + cov.T - np.eye(cov.shape[0])*np.diag(cov)


def proj_spd(cov):
	"""
	finds closest spd matrix (in Frobenius norm sense) to a given cov matrix
	by projecting cov onto the space of spd matrices
	"""

	def is_spd(M):
		"""Returns true when input is positive-definite, via Cholesky"""
		try:
			_ = np.linalg.cholesky(M)
			return True
		except:
			return False
	
	spg = np.linalg.norm(cov)
	_, S, V = np.linalg.svd(cov)

	K = V.T @ np.diag(S) @ V
	cov_new = (cov + K) / 2
	#'symmtetrize':
	cov_new = (cov_new + cov_new.T) / 2

	#so long as not spd repeatedly add smallest eigenvalue
	k = 1

	"""NOTE: this is basically the same kind of "hack" that's applied
	in the make_(...)-functions further down. Should be used with caution.
	"""
	while not is_spd(cov_new):
		mineig = np.min(np.real(np.linalg.eigvals(cov_new)))
		cov_new += np.eye(cov_new.shape[0]) * (-mineig * k**2 + spg)
		k += 1

	return cov_new

def get_eff_overlap_matrix(model, fp):
	"""
	obtains effective recurrent weight matrix under input modulation
	"""

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
	return len(set(pops)) if pops is not None else 1

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

"""
TODO: implement rejection sampling for non-spd covariance matrices
add noise to chols, then sample until matrix is SPD.
"""

def make_MF_covs_pop_dyn(cov_params):

    """
    Create covariance matrices for the model relying on population dynamix

    Args:
        cov_params: dictionary of covariance parameters
    
    Returns:
        covs: covariance matrices

    Here we have:
        - Two populations implementing a counter-clockwise and clockwise rotation
        with frequency w
        - The activity in each population is modulated by the inputs, s.t.
        variance of one input is high on each population -> saturation of non-lin.
        - Covariance matrices are 8d
        -Attention: make sure matrix is spd!
        
    """
    sdM = cov_params["sdM"] #variance of ms (and ns)
    sdR = cov_params["sdR"] #covariance of the diagonal elements -> set radius
    sdOm = cov_params["sdOm"] #covariance of the off-diagonal elements -> set frequency (i.e. clockwise vs. counter-clockwise)
    sdI = cov_params["sdI"] #variance of the respective inputs, one per pop
    sdmW = cov_params["sdmW"] #covariance of the recurrent- and output directions
    sdnI = cov_params["sdnI"] #covariance between the input-selection and input vectors
    sdnW = cov_params["sdnW"] #covariance between the input selection vectors and the ouputs.
    sdW = cov_params["sdW"] #variance of the output vectors (should be roughly OOM as sdM)
    
    eps = 1e-2
    
    #Pop 1:
    cov1 = np.array([[sdM, 0, sdR, sdOm, 0, 0, -sdmW, -sdmW],
                    [0, sdM, -sdOm, sdR, 0, 0, -sdmW, sdmW],
                    [sdR, -sdOm, sdM, 0, sdnI, sdnI, 0, -sdnW],
                    [sdOm, sdR, 0, sdM, sdnI, sdnI, -sdnW, 0],
                    [0, 0, sdnI, sdnI, 0, 0, 0, 0],
                    [0, 0, sdnI, sdnI, 0, sdI, 0, 0],
                    [-sdmW, -sdmW, 0, -sdnW, 0, 0, sdW, 0],
                    [-sdmW, sdmW, -sdnW, 0, 0, 0, 0, sdW]]
                    )
    
    
    cov2 = np.array([[sdM, 0, sdR, -sdOm, 0, 0, sdmW, -sdmW],
                    [0, sdM, sdOm, sdR, 0, 0, sdmW, sdmW],
                    [sdR, sdOm, sdM, 0, sdnI, sdnI, -sdnW, 0],
                    [-sdOm, sdR, 0, sdM, sdnI, sdnI, 0, sdnW],
                    [0, 0, sdnI, sdnI, sdI, 0, 0, 0],
                    [0, 0, sdnI, sdnI, 0, 0, 0, 0],
                    [-sdmW, -sdmW, 0, -sdnW, 0, 0, sdW, 0],
                    [-sdmW, sdmW, sdnW, 0, 0, 0, 0, sdW]]
                    )
    

    #condition matrices to be spd
    if np.linalg.eigh(cov1)[0][0] < 0:
        cov1 -= np.eye(cov1.shape[0])*(np.linalg.eigh(cov1)[0][0] - eps)

    if np.linalg.eigh(cov2)[0][0] < 0:
        cov2 -= np.eye(cov1.shape[0])*(np.linalg.eigh(cov2)[0][0] - eps)
    
    return np.dstack((cov1, cov2))

def make_MF_covs_attr_dyn(cov_params):

	"""
    Create covariance matrices for the model relying on attractor dynamix

    Args:
        cov_params: dictionary of covariance parameters
    
    Returns:
        covs: covariance matrices

    Here we have:
		- A single population with two limit cycle attractors arranged in orthogonal planes
		- Input does not modulate activity 
		- Saturation of the dynamics through the non-linearity comes purely from size of the 
		hidden acitivity (radius in each plane)
        - Covariance matrix is 12-d
        
    """

	eps = cov_params["eps"]
	sdM = cov_params["sdM"] #variance of ms (and ns)
	sdR = cov_params["sdR"] #covariance of the diagonal elements -> set radius
	sdOm = cov_params["sdOm"] #covariance of the off-diagonal elements -> set frequency (i.e. clockwise vs. counter-clockwise)
	sdI = cov_params["sdI"] #variance of the respective inputs, one per pop
	sdmW = cov_params["sdmW"] #covariance of the recurrent- and output directions
	sdnI = cov_params["sdnI"] #covariance between the input-selection and input vectors
	sdnW = cov_params["sdnW"] #covariance between the input selection vectors and the ouputs.
	sdW = cov_params["sdW"] #variance of the output vectors (should be roughly OOM as sdM)
	 

	cov = np.array([[sdM, 0, 0, 0, sdR, -sdOm, 0, 0, 0, 0, eps, eps],
				 	[0, sdM, 0, 0, sdOm, sdR, 0, 0, 0, 0, eps,eps],
					[0, 0, sdM, 0, 0, 0, sdR, sdOm, 0, 0, eps, eps],
					[0, 0, 0, sdM, 0, 0, -sdOm, sdR, 0, 0, eps,eps],
					[sdR, sdOm, 0, 0, sdM, 0, 0, 0, sdnI, 0, 0, 0],
					[-sdOm, sdR, 0, 0, 0, sdM, 0, 0, sdnI, 0, 0, 0],
					[0, 0, sdR, -sdOm, 0, 0, sdM, 0, 0, sdnI, 0, 0],
					[0, 0, sdOm, sdR, 0, 0, 0, sdM, 0, sdnI, 0, 0],
					[0, 0, 0, 0, sdnI, sdnI, 0, 0, sdI, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, sdnI, sdnI, 0, sdI, 0, 0],
					[eps, eps, eps, eps, 0, 0, 0, 0, 0, 0, 0, 0],
					[eps, eps, eps, eps, 0, 0, 0, 0, 0, 0, 0, 0]
					])

	if np.linalg.eigh(cov)[0][0] < 0:
		cov -= np.eye(cov.shape[0])*(np.linalg.eigh(cov)[0][0] - eps)
	
	return cov

def make_MF_covs_LC(cov_params):

	"""
	Covariance matrix that produces an endogenously generated LC

	"""

	eps = 1e-4

	sdM = cov_params["sdM"]
	sdR = cov_params["sdR"]
	sdOm = cov_params["sdOm"]

	cov = np.array([[sdM, 0, sdR, sdOm, 0, 0],
				 	[0, sdM, -sdOm, sdR, 0, 0],
					[sdR, -sdOm, sdM, 0, 0, 0],
					[sdOm, sdR, 0, sdM, 0, 0],
					[0, 0, 0, 0, eps, 0],
					[0, 0, 0, 0, 0, eps]
					])
	
	if np.linalg.eigh(cov)[0][0] < 0:
		cov -= np.eye(cov.shape[0])*(np.linalg.eigh(cov)[0][0] - eps)
	
	return cov

def make_MF_covs_LC3D(cov_params):

	"""
	Covariance matrix that produces an endogenously generated LC

	"""

	eps = 1e-4

	sdM = cov_params["sdM"]
	sdR = cov_params["sdR"]
	sdOm = cov_params["sdOm"]

	cov = np.array([[sdM, 0, 0, sdR, sdOm, 0, 0, 0],
				 	[0, sdM, 0, -sdOm, sdR, 0,  0, 0],
					[0, 0, sdM, 0, 0, 0, 0, 0],
					[sdR, -sdOm, 0, sdM, 0, 0, 0, 0],
					[sdOm, sdR, 0, 0, sdM, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, eps, 0],
					[0, 0, 0, 0, 0, 0, 0, eps]
					])
	
	if np.linalg.eigh(cov)[0][0] < 0:
		cov -= np.eye(cov.shape[0])*(np.linalg.eigh(cov)[0][0] - eps)
	
	return cov

def make_MF_covs_RA(cov_params):
	"""
	Covariance matrix that produces endogenoluslt generated RA
	"""
	#simply do it by setting sdOM to 0 and calling make_MF_covs_LC
	cov_params["sdOm"] = 0
	return make_MF_covs_LC(cov_params)
	
	
def make_MF_covs_attr_inhib (cov_params):

	"""
    Create covariance matrices for the model relying on attractor dynamix
	mutual inhibition stabilises attractors 

    Args:
        cov_params: dictionary of covariance parameters
    
    Returns:
        covs: covariance matrices

    Here we have:
		- A single population with two limit cycle attractors arranged in orthogonal planes
		- Input does not modulate activity 
		- Saturation of the dynamics through the non-linearity comes purely from size of the 
		hidden acitivity (radius in each plane) and mutual inhibition
        - Covariance matrix is 12-d
        
    """

	eps = 1e-2

	sdM = cov_params["sdM"] #variance of ms (and ns)
	sdR = cov_params["sdR"] #covariance of the diagonal elements -> set radius
	sdOm = cov_params["sdOm"] #covariance of the off-diagonal elements -> set frequency (i.e. clockwise vs. counter-clockwise)
	sdI = cov_params["sdI"] #variance of the respective inputs, one per pop
	sdmW = cov_params["sdmW"] #covariance of the recurrent- and output directions
	sdnI = cov_params["sdI"] #covariance between the input-selection and input vectors
	sdnW = cov_params["sdnW"] #covariance between the input selection vectors and the ouputs.
	sdW = cov_params["sdW"] #variance of the output vectors (should be roughly OOM as sdM)
	sd_inhib = cov_params["sdIhib"]
	 

	cov = np.array([[sdM, 0, 0, 0, sdR, -sdOm, sd_inhib, 0, 0, 0, eps, eps],
				 	[0, sdM, 0, 0, sdOm, sdR, 0, sd_inhib, 0, 0, eps,eps],
					[0, 0, sdM, 0, sd_inhib, 0, sdR, sdOm, 0, 0, eps, eps],
					[0, 0, 0, sdM, 0, sd_inhib, -sdOm, sdR, 0, 0, eps,eps],
					[sdR, sdOm, sd_inhib, 0, sdM, 0, 0, 0, sdnI, 0, 0, 0],
					[-sdOm, sdR, 0, sd_inhib, 0, sdM, 0, 0, sdnI, 0, 0, 0],
					[sd_inhib, 0, sdR, -sdOm, 0, 0, sdM, 0, 0, sdnI, 0, 0],
					[0, sd_inhib, sdOm, sdR, 0, 0, 0, sdM, 0, sdnI, 0, 0],
					[0, 0, 0, 0, sdnI, sdnI, 0, 0, sdI, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, sdnI, sdnI, 0, sdI, 0, 0],
					[eps, eps, eps, eps, 0, 0, 0, 0, 0, 0, 0, 0],
					[eps, eps, eps, eps, 0, 0, 0, 0, 0, 0, 0, 0]
					])
	"""
	if np.linalg.eigh(cov)[0][0] < 0:
		cov -= np.eye(cov.shape[0])*(np.linalg.eigh(cov)[0][0] - eps)
	"""
	return cov

def extract_idxs(model):

	r = model.rnn.rank
	in_dim = model.rnn.in_dim
	nidx = r
	Iidx = 2 * r
	Widx = 2 * r + in_dim

	return nidx, Iidx, Widx


def compute_gammas(model, inputs):
    rates, out = run_rnn(model, inputs)
    gammas = phiPrime(rates)
    return gammas

def compute_deltas(model, k):
	"""
	computes deltas of populations for MeanFieldRNNCells
	"""
	
	covs = model.rnn.covs.detach().numpy() 
	

	K = k[:,:model.rnn.rank]
	V = k[:,model.rnn.rank:]
	
	nidx, Iidx, Widx = extract_idxs(model)
	deltas = np.sum(
				np.hstack(
					(
					np.einsum('iik, li -> lik', covs[Iidx:Widx, Iidx:Widx,:], V**2),
					np.einsum('iik, li -> lik', covs[:nidx, :nidx, :], K**2)
					), 
				), axis=1
			)
	
	return deltas

def g(deltas):
	"""Gaussian integral"""
	return 1 / np.sqrt(1 + (np.pi / 2) * deltas)
	
def gPrime(deltas):
	"""Derivative of g"""
	return -(np.pi / 2) * g(deltas)**3

def compute_fixed_point_simple(model,
							   in_,
							   xlims=[-1,1],
							   ylims=[-1,1],
							   trnst=0,
							   num_fp_trials=20,
							   dist_thresh=1e-2
							   ):
	"""
	alternative to the fp-finder class for really low-d models (i.e. meanfields)
	"""
	#TODO: integrate this into plotting func.
	from scipy.optimize import root

	def dkdt(k):
    
		if len(k.shape)==1:
			k = k.reshape(1, k.shape[0])

		K = k[:,:2]
		V = k[:,2:]
		rec_input = np.stack(
							(np.einsum('...ijk, ...lj -> ...lik', covs[nidx:Iidx, :nidx,:], K),
							np.einsum('...ijk, ...lj -> ...lik', covs[nidx:Iidx, Iidx:Widx,:], V) 
							)
						)

		gains = g(compute_deltas(model, k))
		rec_input = (rec_input * weights * gains).sum(axis=(0,3))
		dkdt = -K + rec_input
		return dkdt
	
            
	def DKDT(k, input):
		"""
		This is the function passed to the root solver
		"""

		k_full = torch.from_numpy(
					np.hstack((
						np.expand_dims(k, axis=0), 
						np.expand_dims(input, axis=0)
						))
					).to(torch.float32)
		dkdt_ = dkdt(k_full)
		return dkdt_[0,:2].detach().numpy()

	X_grid, Y_grid = np.meshgrid(
							np.linspace(xlims[0], xlims[1], int(np.sqrt(num_fp_trials))),
							np.linspace(ylims[0], ylims[1], int(np.sqrt(num_fp_trials)))
							)
	fps = []
	for i in range(X_grid.ravel().size):
		x0 = np.array([X_grid.ravel()[i], Y_grid.ravel()[i]])
		#again, assume input equilibriates
		sol = root(DKDT, x0=x0, args=input[trnst,:])
		if all([np.linalg.norm(sol.x - other.x) > dist_thresh for other in fps]):
			fps.append(sol)

	jacs = []
	for fp in fps:
		k = fp.x
		jac = compute_jacobian_mf_analytical(model, k, in_)
		jacs.append(jac)

	return fps, jacs
	
def compute_jacobian_mf_analytical(model, k, in_):
	"""
	return the jacobian of the model dynamics at k, with input in_
	Args:
		model: the model
		k: np.array[rank] -> the fixed point 
		in_: np.array[in_dim] -> the input
	Return:
		S: np.array[rank x rank] -> the jacobian

	"""
	nidx, Iidx, Widx = extract_idxs(model)
	#this is just for the delta computation
	k_ = np.hstack((np.expand_dims(k, axis=0),
               np.expand_dims(in_, axis=0))
			)

	covs = model.rnn.covs.detach().numpy()
	weights = model.rnn.weights.unsqueeze(dim=0).detach().numpy().astype(np.float64)
	deltas = compute_deltas(model, k_)
	S = -np.eye(model.rnn.rank)
	S += (covs[nidx:Iidx, :nidx, :] * g(deltas) * weights).sum(axis=2)

	#Correction term
	sk = (np.einsum('ijk, j -> ik', covs[nidx:Iidx, :nidx, :], k)
		+ np.einsum('ijk, j -> ik', covs[nidx:Iidx, Iidx:Widx], in_)
		) 
	sk += np.einsum('iik, i -> ik', covs[:nidx, :nidx, :], k)
	S += (sk * gPrime(deltas)**3 * weights)

	return S.sum(axis=2)

#===============MISCELLANEOUS===================

def build_state_dicts_from_params(params, idx):
    from collections import OrderedDict
    state_dict = OrderedDict(
                ("rnn." + name, torch.from_numpy(param))
                for name, param in params[idx].items()
)
    return state_dict

def reset_model_from_params_idx(model, params, idx):
	"""
	resets model to specific configuration 
	mostly used for analysing model dynamics under training
	"""
	state_dict = build_state_dicts_from_params(params, idx)
	model.load_state_dict(state_dict, strict=False)

def compute_rate_similarity(rates, base_idx, trnst):

	base_idx = 0
	trnst = 0
	_, S_base, _ = np.linalg.svd(rates[base_idx, trnst:,:])
	sims = []

	for idx in range(rates.shape[0]):
		U, S, V = np.linalg.svd(rates[idx,trnst:,:])
		sim = (np.linalg.norm(rates[base_idx,...] @ rates[idx, ...].T) 
		 		/ np.linalg.norm(np.multiply(S, S_base))
			)
		sims.append(sim)

	return sims




				    


	



















