
import numpy as np
import torch 
from torch.utils.data import Dataset

NUM_TRIALS = 50


class PSineWaveUpDown(Task):

    """Pulsed Sinewave Task with only two trained frequencies 
    to match the shape of the Pulsed sdcycling task"""

    
    def get_stim(self, idx):

        label = 1 if idx >= np.floor(self.__len__()/2) else 0

        trial_len = self.task_params["trial_len"]
        in_dim = self.task_params["in_dim"]
        delay = self.task_params["rand_delay"]

        delay = torch.randint(1, delay, size=(1,)) if delay else torch.tensor(0)

        trial = torch.zeros((trial_len, in_dim), dtype=torch.float32)
        trial[delay, label] = 1/self.task_params["dt"]

        return trial, label, delay.item()

    def get_target(self, idx):

        freqs = self.task_params["freq"]
        amp = self.task_params["amp"]
        phase = self.task_params["phase"]

        trial_len = self.task_params["trial_len"]
        out_dim = self.task_params["out_dim"]
        target = torch.zeros((trial_len, out_dim), dtype=torch.float32)
        mask = target.detach().clone().to(dtype=bool)

        _, label, delay = self.get_stim(idx)
        t = torch.linspace(delay, trial_len, trial_len-delay)
        target[delay:trial_len, 0] = amp * torch.sin(2*torch.pi*freqs[label]*t)
        target[delay:trial_len, 1] = amp * torch.cos(2*torch.pi*freqs[label]*t)
        mask[delay:trial_len,:] = 1

        if not self.task_params["with_cosine"]:
            mask[delay:trial_len,1] = 0

        return target, mask