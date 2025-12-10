"""SineWave Dataset: SineWaves created as in https://dl.acm.org/doi/10.1162/NECO_a_0040 """
import torch, numpy as np
from torch.utils.data import Dataset
from Tasks.TaskManager import Task


class SineWaveUpDown(Task):

    """Sinewave Task with only two trained frequencies to match the shape of the cycling task"""

    def __name__():
        return "SineWave"

    def get_stim(self, idx):

        label = 1 if idx >= np.floor(self.__len__()/2) else 0

        trial_len = self.task_params["trial_len"]
        in_dim = self.task_params["in_dim"]
        delay = self.task_params["rand_delay"]

        delay = torch.randint(1, delay, size=(1,)) if delay else torch.tensor(0)

        trial = torch.zeros((trial_len, in_dim), dtype=torch.float32)

        #set input
        trial[delay:trial_len,label] = 1

        return trial, label, delay.item()

    def get_target(self, idx):

        freqs = self.task_params["freqs"]
        amp = self.task_params["amp"]
        phase = self.task_params["phase"]

        trial_len = self.task_params["trial_len"]
        out_dim = self.task_params["out_dim"]
        target = torch.zeros((trial_len, out_dim), dtype=torch.float32)
        mask = target.detach().clone().to(dtype=bool)

        step = .01
        _, label, delay = self.get_stim(idx)
        t = torch.linspace(delay, trial_len*step, trial_len-delay)
        target[delay:trial_len, 0] = amp * torch.sin(2*torch.pi*freqs[label]*t)
        target[delay:trial_len, 1] = amp * torch.cos(2*torch.pi*freqs[label]*t)
        mask[delay:trial_len,:] = 1

        if not self.task_params["with_cosine"]:
            mask[delay:trial_len,1] = 0

        return target, mask

    def generate_inputs(self, num_inputs):

        trial, _, _ = self.get_stim(idx=0)
        antitrial, _, _ = self.get_stim(idx=self.__len__())

        #draw interpolated inputs from upper right quadrant
        angles = np.linspace(0, np.pi/2, num_inputs)
        inputs = torch.zeros((num_inputs, *trial.shape))

        for idx, phi in enumerate(angles):
            const_input_tensor = const_input_tensor = np.ones_like(trial)*np.array([np.cos(phi), np.sin(phi)])
            const_input_tensor = torch.tensor(const_input_tensor).float()
            inputs[idx,:,:] = const_input_tensor

        return inputs
