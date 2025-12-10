"""
Module for fixpoint logic
"""

import torch, numpy as np
from torch.autograd import Variable, functional
import os, pickle as pkl
from rnn import RNN as RNN
from typing import Union, Optional, NewType, Literal, List
from dataclasses import dataclass

TensorArray = NewType('TensorArray', Union[torch.tensor, np.ndarray])
FixedPointType = NewType('FixedPointType', Literal["Stable Spiral", 
												"Unstable Spiral",
												"Saddle",
												"Stable FP",
												"Unstable FP"
												   ])


@dataclass
class FixedPoint:

	"""
	Dataclass for storing single fixpoints
	"""

	model: RNN
	input_: torch.tensor
	rates: torch.tensor
	x_star: torch.tensor
	x_init: torch.tensor
	F_xstar: torch.tensor
	qstar: Union[torch.float64, torch.float32]
	dqstar: Union[torch.float64, torch.float32]
	Jdu_star: torch.tensor
	Jdx_star: torch.tensor
	_check: bool


	def __init__(self,
				model: Optional[RNN],
				input_: torch.tensor,
				rates: torch.tensor,
				x_star: torch.tensor,
				x_init: torch.tensor,
				F_xstar: torch.tensor,
				qstar: Union[torch.float64, torch.float32],
				dqstar: Union[torch.float64, torch.float32],
				Jdu_star: torch.tensor,
				Jdx_star: torch.tensor,
				_check: bool,
				hiddens: List
		):

		self.model = model
		self.input_ = input_
		self.rates = rates
		self.x_star = x_star
		self.x_init = x_init
		self.F_xstar = F_xstar
		self.qstar = qstar
		self.dqstar = dqstar
		self.Jdu_star = Jdu_star
		self.Jdx_star = Jdx_star
		self._check = _check
		self.hiddens = hiddens


	def calc_jacobians_xstar(self):

		fp = Variable(self.x_star.unsqueeze(dim=0).type_as(self.input_))
		fp.requires_grad = True

		#define the flow
		def F(input_, hidden):
			rates, _ = self.model.rnn(input_, hidden)
			return rates

		jacobians = functional.jacobian(F,(self.input_.unsqueeze(dim=0), fp))

		Jdu, Jdx = jacobians
		Jdx = Jdx.squeeze()[0,:,:].detach()
		Jdu = Jdu.squeeze().detach()

		self.Jdx_star = Jdx
		self.Jdu_star = Jdu

	def calc_F_xstar(self, input_: torch.Tensor):

		nonlinearity = self.model.rnn.nonlinearity
		w_rec = self.model.rnn.w_rec
		input_= self.input_ @ self.model.w_in

		def dxdt(x, input_=input_):
			dxdt = -x + w_rec @ nonlinearity(x) + input_
			return dxdt

		return dxdt(self.x_star, input_[0,:])

	def decompose_Jdx(self, dim=3):
		pass

	@classmethod
	def _setup(cls, model, input_):
		"""
		Initialize an empty fix point object
		"""

		size = model.rnn.net_size
		trial_len, input_dim = input_.shape
		rates, _ = model(input_.unsqueeze(dim=0))

		xstar_tensor = torch.zeros((3, size))
		xstar_tensor.fill_(float('nan'))
		x_star, x_init, F_xstar = xstar_tensor
		qstar = float('nan')
		dqstar = float('nan')
		Jdx_star = torch.zeros((size, size))
		Jdx_star.fill_(float ('nan'))
		Jdu_star = torch.zeros((trial_len, size, trial_len, input_dim))
		Jdu_star.fill_(float('nan'))
		eigvals = np.zeros(size)
		eigvecs = np.zeros((size))
		eigvals[:] = np.nan
		eigvecs[:] = np.nan
		Jdx_star_decomp = (eigvals, eigvecs)
		_check = False
		hiddens = []

		return cls(model,
				input_,
				rates,
				x_star,
				x_init,
				F_xstar,
				qstar,
				dqstar,
				Jdx_star,
				Jdu_star,
				_check,
				hiddens)

	@classmethod
	def fps2pickle(cls, fps, path, model_path: Optional[str]=None):

		"""
		hacky way of getting around no lambda pickle snafu 
		-> TODO: find better way -> dill?
		"""
		if not model_path:
			model_path = fps[0].model.params['path']
		#remove model attr -> will be stored with model, so you always have that
		[fp.__setattr__('model', None) for fp in fps]
		fps_path = os.path.join(model_path, path)
		if not os.path.isdir(fps_path):
			os.mkdir(fps_path)
		with open(os.path.join(fps_path, "fps.pkl"), "wb") as fps_file:
			pkl.dump(fps, fps_file)




								

	