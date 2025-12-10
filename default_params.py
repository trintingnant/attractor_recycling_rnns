def default_params():

	#Default params for Cycling task
	#Low rank RNN loadings:

	model_params = {
		"nonlinearity": "tanh",								#activation function for hidden units
		"out_nonlinearity": "identity",						#activation function for output units
		"input_size": 2,									#number of input units
		"recurrent_size": 100,								#number of hidden units
		"output_size": 2, 									#number of output units
		"init_scale": .1,									#scaling of initial state
		"dt": .5,											#update step
		"tau": 1.,											#time constant
		"w_in_scale": 1,									#scaling of input weights
		"w_out_scale": 1,									#scaling of output weights
		"spectral_norm": True,								#spectral normalisation?
		"spectral_radius": 1.5,								#spectral radius of recurrent weight matrix
		"1overN_scale": True,								#scale output by 1/N rather than 1/sqrt(N)
		"noise_amplitude": .01,								#noise intensity
		"low_rank": False,									#low_rank network?
		"rank": 3, 											#rank for low_rank matrices
		"low_rank_cov": .6,									#correlation betweeen left and right eigenvectors m and n
		"LR_method": "cholesky",							#method for computing m and n
		"lr_cov": None,										#provide covariance matrix directly for m, n computation
		"lr_w_out_std": 4.5,								#kind of hacky: sets w_out_std higher if needed for lr case
		"train_w_out": False,								#train output vector?			
		"train_w_in": True,									#train input vector?
		"multi_pop_gmm": False								#use GMMLowRankNet
	
	}

	task_params = {
		"num_trials": 2,									#number of trials
		"trial_len": 200,									#length of single trial
		"in_dim": 2,										#input dimension
		"rand_delay": 20,									#delay interval (~U[0...rand_delay])
    	"rand_shutoff": 40,									#stimulus shutoff (~U[rand_shutoff...trial_len])
    	"out_dim": 2,										#output dimension
    	"num_rotations": 4,									#number of rotations (i.e. frequency of sine and cosines)
    	"freq": 1,
    	"amp": 1,
    	"phase": 0,
    	"with_cosine": True									#Use cosine as second output dimension?
	}

	training_params = {
		"batch_size": 1,									#minibatch
		"shuffle_dataset": True,							#whether to shuffle the dataset
		"loss_func": "mse",									#loss function
		"optimizer": "adam",								#optimizer
		"n_epochs": 100,									#number of training epochs
		"lr": 1e-2,											#learning rate
		"loss_tol": .01										#tolerated loss						

	}


	return model_params, task_params, training_params


def fpf_params():

	fpf_params = {

		"gamma": 1e-01,										#step size
		"speed_tor": [1e-04, 1e-06],						#speed tolerance
		"d_speed_tor": 1e-04,								#acc tolerance
		"max_epochs": 50000,								#max number of epochs
		"lr_decay_epoch": 2500,								#if using simple exp decay: no. epochs after which to anneal
		"lr_decay_factor": .5,								#if using sinple exp decay: factor by which to anneal
		"noise_scale": 0.0,									#noise scale for IC sampling
		"random_seed": 0 									#random seed for IC sampling

	}

	scheduler_params = {

		"mode": 'min',
		"patience": 3,
		"cooldown": 0

	}

	return fpf_params, scheduler_params
	




	


