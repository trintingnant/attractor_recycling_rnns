import torch, torch.nn as nn, numpy as np
import pickle as pkl, os, sys, time
from typing import Optional, Union, Tuple, List
from torch.utils.data import DataLoader
from rnn import RNN, save_rnn, load_rnn
from utils import mse, zero_loss
import os, uuid

#Okay: test, test, test ... remote, test... 

PARAMS_2_LOG = ['w_rec', 'w_in']
RAW_LOG_FREQ = 5


def train_rnn(model: RNN,
				task,
				training_params,
				init_hidden: Optional[torch.Tensor]=None,
				float_precision=False,
				model_params: Optional[dict]=None,
				task_params: Optional[dict]=None,
				verbose=True,
				log_raw_data=False,
				use_wandb=True
				) -> Union[List, Tuple[List, List, List]]:

	if use_wandb:

		try:
			import wandb
			WANDB_PARAMS = {"use_wandb": True, "wandb_freq": 10}
			wandb.login()

		except:
			WANDB_PARAMS = {}
			print('wandb not configured...')

	else:

		wandb_params = {}

	#if logging make sure to pass all files
	if log_raw_data:
		assert model_params is not None
		assert task_params is not None

	dataloader = DataLoader(task, 
		batch_size=training_params["batch_size"], 
		shuffle=training_params["shuffle_dataset"])

	n_epochs = training_params["n_epochs"]
	display_epochs = np.floor(n_epochs / 25)

	try:
		loss_func = training_params.get("loss_func")

		if loss_func == "mse":
			loss_func = mse
	except:
		#defaults to zero loss
		loss_func = zero_loss

	if training_params["optimizer"] == "adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=training_params["lr"])

	#learning rate scheduler
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer,
		T_max=training_params["n_epochs"]
		)

	#synchronise with wandb for logging:
	if wandb_params:
		wandb.init(
			project='attractor_sharing',
			group='pytorch',
			config={**model.params, 
					**dataloader.dataset.task_params, 
					**training_params
					}
				)
		
		#SUPPRESS auto-saves for now: clogs up HD
		"""
		#intialise file with run name if logged to wandb
		run_name = wandb.run.name
		save_rnn(f"saved_models/{run_name}",
		  	model,
		  	model_params,
		  	task_params,
		  	training_params
		  	)
		"""

		config = wandb.config
		#Attention: `watch` only records histograms -> (almost entirely) useless
		#wandb.watch(model.rnn, log="all", log_freq=wandb_params["wandb_freq"])
		print("wandb_run:", wandb.run)
	"""
	else:
		#generate random string from uuid
		id = str(uuid.uuid4())[:10]
		model_path = f"saved_models/{id}"
		save_rnn(model_path,
		   		model,
				model_params,
				task_params,
				training_params
				)
	"""
	if log_raw_data:
		grads = []
		params = []

	reg_fns = []
	reg_lambdas = []
	if training_params["use_l1_reg"]:
		reg_fns.append(l1_loss)
		reg_lambdas.append(training_params["lambda_l1"])

	if training_params["use_l2_reg"]:
		reg_fns.append(l2_loss)
		reg_lambdas.append(training_params["lambda_l2"])

	if training_params["use_l1_cov_reg"]:
		#only makes sense for MF models
		assert(model.params["mean_field"]==True)
		reg_fns.append(l1_cov_loss)
		reg_lambdas.append(training_params["lambda_l1_cov"])

	if training_params["use_l2_cov_reg"]:
		#only makes sense for MF models
		assert(model.params["mean_field"]==True)
		reg_fns.append(l2_cov_loss)
		reg_lambdas.append(training_params["lambda_l2_cov"])

	start_time = time.time()
	loss_prev = np.infty
	model.train() 

	losses = []
	reg_losses = []

	for i in range(n_epochs):

		loss_epoch = .0
		num_len = 0

		if loss_prev <= training_params["loss_tol"]:
			break

		for x, y, m in dataloader:

			#forward pass
			if float_precision:
				rates, y_pred = model(x.float(), init_hidden)
				
			else: 
				rates, y_pred = model(x, init_hidden)

			#clear optimizer
			optimizer.zero_grad()

			#gradient descent
			loss = loss_func(y_pred, y, m)
			print(len(reg_fns))
			if len(reg_fns) != 0:
				reg_losses = torch.tensor(
					[reg_fn(rates[:,1:], covs=model.rnn.covs)
	 				for reg_fn in reg_fns
					]
				)
				loss += (reg_losses * reg_lambdas).sum()
				
			loss.backward()

			#log `raw` (!!!) grads and params every x steps
			if log_raw_data :
				if i % RAW_LOG_FREQ == 0:

					with torch.no_grad():

						#TODO: improve this
						grads.append({
							name:param.grad.numpy()
							for name, param in model.rnn.named_parameters() \
							if param.grad is not None
							}
						)

						params.append({
							name:param.detach().clone().numpy()
							for name, param in model.rnn.named_parameters() \
							}
						)
			
			optimizer.step()
			scheduler.step()
			num_len += 1
			loss_epoch += loss.item()
		
		#batch loss
		loss_epoch /= num_len
		loss_prev = loss_epoch
		if wandb_params:
			wandb.log({"loss": loss_epoch})
		losses.append(loss_epoch)

		if verbose and i % display_epochs == 0 and i != 0:
	
			print(
					f"epoch {i+1 :d} / {n_epochs}, \
					time={time.time()-start_time :.5f}s, \
					task_loss={loss_epoch}"
					)
		
	print(f"\nDone: training took {time.time()-start_time:.1f} sec.")
	model.eval()

	if wandb_params:

		#manually removing wandb hooks, because `unwatch` not working. WHY?
		wandb.finish()

	if log_raw_data:
		return losses, grads, params
	else:
		return losses
	
	#NOTE: this logging method is more or less whack. Can build logger?


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







