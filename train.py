import torch, torch.nn as nn, numpy as np
import pickle as pkl, os, sys, time
from typing import Optional, List, Callable, Dict
from torch.utils.data import DataLoader
from rnn import RNN
from utils import mse, zero_loss, build_regularisers
import time
from pathlib import Path

class TrainingLogger:

	"""
	Logger class for lightweight logging 
	
	"""

	def __init__(
		self,
		log_freq: int = 1,
		log_initial: bool = True,
	):
		self.log_freq = log_freq
		self.log_initial = log_initial

		self.callbacks: Dict[str, Callable] = {}
		self.gradient_callbacks: Dict[str, Callable] = {}

		self.history: Dict[str, list] = {}

		self.epochs = []
		self.steps = []

		self.grad_epochs = []
		self.grad_steps = []

	def add(
		self,
		name: str,
		callback: Callable,
	):
		self.callbacks[name] = callback

		if name not in self.history:
			self.history[name] = []

	# quick & dirty: log single grad
	def add_gradient(
		self,
		name: str,
		callback: Callable,
	):
		self.gradient_callbacks[name] = callback

		if name not in self.history:
			self.history[name] = []
	
	# quick & dirty: log all model params
	def add_parameters(
		self,
		name="params",
	):
		def callback(model, **kwargs):
			return {
				param_name: param.detach().cpu().clone()
				for param_name, param in model.rnn.named_parameters()
			}

		self.add(name, callback)

	# quick & dirty: log ALL gradients
	def add_gradients(
		self,
		name="grads",
	):
		def callback(model, **kwargs):
			return {
				param_name: (
					param.grad.detach().cpu().clone()
					if param.grad is not None
					else None
				)
				for param_name, param in model.rnn.named_parameters()
			}

		self.add_gradient(name, callback)

	def _copy_value(
		self,
		value,
	):
		if torch.is_tensor(value):
			return value.detach().cpu().clone()

		if isinstance(value, np.ndarray):
			return value.copy()

		if isinstance(value, dict):
			return {
				key: self._copy_value(val)
				for key, val in value.items()
			}

		if isinstance(value, list):
			return [
				self._copy_value(val)
				for val in value
			]

		if isinstance(value, tuple):
			return tuple(
				self._copy_value(val)
				for val in value
			)

		return value

	def log(
		self,
		model,
		epoch,
		step,
		force=False,
		**kwargs,
	):
		if not force and epoch % self.log_freq != 0:
			return

		self.epochs.append(epoch)
		self.steps.append(step)

		with torch.no_grad():
			for name, callback in self.callbacks.items():
				value = callback(
					model=model,
					epoch=epoch,
					step=step,
					**kwargs,
				)

				self.history[name].append(
					self._copy_value(value)
				)

	def log_gradients(
		self,
		model,
		epoch,
		step,
		**kwargs,
	):
		if epoch % self.log_freq != 0:
			return

		self.grad_epochs.append(epoch)
		self.grad_steps.append(step)

		for name, callback in self.gradient_callbacks.items():
			value = callback(
				model=model,
				epoch=epoch,
				step=step,
				**kwargs,
			)

			self.history[name].append(
				self._copy_value(value)
			)

	def state_dict(self):
		return {
			"log_freq": self.log_freq,
			"log_initial": self.log_initial,
			"history": self.history,
			"epochs": self.epochs,
			"steps": self.steps,
			"grad_epochs": self.grad_epochs,
			"grad_steps": self.grad_steps,
		}

	def load_state_dict(
		self,
		state,
	):
		self.log_freq = state["log_freq"]
		self.log_initial = state["log_initial"]
		self.history = state["history"]
		self.epochs = state["epochs"]
		self.steps = state["steps"]
		self.grad_epochs = state.get("grad_epochs", [])
		self.grad_steps = state.get("grad_steps", [])

		return self

	def save(
		self,
		path,
	):
		path = Path(path)

		if path.exists() and path.is_dir():
			path = path / "logger.pt"

		elif path.suffix == "":
			path = path.with_suffix(".pt")

		path.parent.mkdir(parents=True, exist_ok=True)

		torch.save(
			self.state_dict(),
			path,
	)

	@classmethod
	def load(
		cls,
		path,
	):
		state = torch.load(
			path,
			map_location="cpu",
			weights_only=False,
		)

		logger = cls(
			log_freq=state["log_freq"],
			log_initial=state["log_initial"],
		)

		logger.load_state_dict(state)

		return logger

	def __getitem__(
		self,
		key,
	):
		return self.history[key]

	def keys(self):
		return self.history.keys()


def train_rnn(
	model: RNN,
	task,
	training_params,
	init_hidden: Optional[torch.Tensor]=None,
	float_precision=False,
	verbose=True,
	use_wandb=False,
	retain_graph=False,
	logger: Optional[TrainingLogger]=None
) -> List:

	if use_wandb:

		try:
			import wandb

			wandb_params = {
				"use_wandb": True,
				"wandb_freq": 10
			}

			wandb.login()

		except:
			wandb_params = {}
			print("wandb not configured...")

	else:

		wandb_params = {}

	dataloader = DataLoader(
		task,
		batch_size=training_params["batch_size"],
		shuffle=training_params["shuffle_dataset"]
	)

	n_epochs = training_params["n_epochs"]

	try:
		loss_func = training_params.get("loss_func")

		if loss_func == "mse":
			loss_func = mse

	except:
		loss_func = zero_loss

	if training_params["optimizer"] == "adam":

		optimizer = torch.optim.Adam(
			model.parameters(),
			lr=training_params["lr"]
		)

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer,
		T_max=training_params["n_epochs"]
	)

	if wandb_params:

		wandb.init(
			project="attractor_sharing",
			group="pytorch",
			config={
				**model.params,
				**dataloader.dataset.task_params,
				**training_params
			}
		)

		config = wandb.config

		print(
			"wandb_run:",
			wandb.run
		)

	regs = [
		training_params.get("use_l1_reg", False),
		training_params.get("use_l2_reg", False),
		training_params.get("use_l1_cov_reg", False),
		training_params.get("use_l2_cov_reg", False)
	]

	reg_lambdas = [
		training_params.get("lambda_l1", 0.0),
		training_params.get("lambda_l2", 0.0),
		training_params.get("lambda_l1_cov", 0.0),
		training_params.get("lambda_l2_cov", 0.0)
	]

	reg_fns = build_regularisers(
		regs
	)

	start_time = time.time()
	loss_prev = np.infty

	model.train()

	losses = []
	reg_losses = []

	global_step = 0

	for i in range(n_epochs):

		loss_epoch = .0
		num_len = 0

		if loss_prev <= training_params["loss_tol"]:
			break

		for x, y, m in dataloader:

			if float_precision:

				rates, y_pred = model(
					x.float(),
					init_hidden
				)

			else:

				rates, y_pred = model(
					x,
					init_hidden
				)

			optimizer.zero_grad()

			loss = loss_func(
				y_pred,
				y,
				m
			)

			if len(reg_fns) != 0:

				reg_losses = torch.tensor(
					[
						reg_fn(
							rates[:,1:],
							covs=model.rnn.covs
						)
						for reg_fn in reg_fns
					]
				)

				loss += (
					reg_losses * reg_lambdas
				).sum()

			loss.backward(
				retain_graph=retain_graph
			)

			# log gradients here, before they are flushed after .step()
			if logger is not None:

				logger.log_gradients(
					model=model,
					epoch=i,
					step=global_step,
					loss=loss,
					rates=rates,
					y_pred=y_pred,
					x=x,
					y=y,
					mask=m,
					optimizer=optimizer
				)

			optimizer.step()

			global_step += 1
			num_len += 1
			loss_epoch += loss.item()

		scheduler.step()
		loss_epoch /= num_len
		loss_prev = loss_epoch

		# log params here
		if logger is not None:

				logger.log(
					model=model,
					epoch=i,
					step=global_step,
					loss=loss,
					rates=rates,
					y_pred=y_pred,
					x=x,
					y=y,
					mask=m,
					optimizer=optimizer
				)

		if wandb_params:

			wandb.log({
				"loss": loss_epoch
			})

		losses.append(
			loss_epoch
		)

		if verbose:

			print(
				f"epoch {i+1:d} / {n_epochs}\t"
				f"time={time.time()-start_time:.5f}s\t"
				f"task_loss={loss_epoch}",
				end="\r"
			)

	print(
		f"\nDone: training took "
		f"{time.time()-start_time:.1f} sec."
	)

	model.eval()

	if wandb_params:
		wandb.finish()

	return losses












