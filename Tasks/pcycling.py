import torch, numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append('../Tasks')
from Tasks.TaskManager import Task


class PCycling(Task):

	def __name__():
		return "PCycling"

	#define stimuli
	def get_stim(self, idx):

		label = 1 if idx >= np.floor(self.__len__()/2) else 0

		trial_len = self.task_params["trial_len"]
		in_dim = self.task_params["in_dim"]
		delay = self.task_params["rand_delay"]

		delay = torch.randint(1, delay, size=(1,)) if delay else torch.tensor(0)

		trial = torch.zeros((trial_len, in_dim), dtype=torch.float32)
		trial[delay, label] = 1/self.task_params['dt']

		return trial, label, delay.item()


	def get_target(self, idx):

		freq = self.task_params["freq"]
		amp = self.task_params["amp"]
		phase = self.task_params["phase"]

		trial_len = self.task_params["trial_len"]
		out_dim = self.task_params["out_dim"]
		target = torch.zeros((trial_len, out_dim), dtype=torch.float32)
		mask = target.detach().clone().to(dtype=bool)

		_, label, delay = self.get_stim(idx)
		categorical_label = 1 if label==1 else -1
		t = torch.linspace(delay, trial_len, trial_len-delay)
		target[delay:trial_len, 0] = categorical_label * torch.sin(2*torch.pi*freq*t)
		target[delay:trial_len, 1] = amp * torch.cos(2*torch.pi*freq*t)
		mask[delay:trial_len,:] = 1

		if not self.task_params["with_cosine"]:
			mask[delay:trial_len-1, 1] = 0

		return target, mask

	def generate_inputs(self, num_inputs):

		trial, _, _ = self.get_stim(idx=0)
		antitrial, _, _ = self.get_stim(idx=self.__len__())
		d = self.task_params["rand_delay"]

		#draw interpolated inputs from upper right quadrant
		angles = np.linspace(0, np.pi/2, num_inputs)
		inputs = torch.zeros((num_inputs, *trial.shape))

		for idx, phi in enumerate(angles):
			const_input_tensor = const_input_tensor = np.zeros_like(trial)*np.array([np.cos(phi), np.sin(phi)])
			const_input_tensor[d,:] = torch.tensor(const_input_tensor).float()
			inputs[idx,:,:] = const_input_tensor

		return inputs




