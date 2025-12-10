"""
this is an implmentation of the SET-card game task,
a simple arithmetic binary classification task inspired by Powers (2022) and Murray (2023)
Murray (2023) especially claims that RNNs solving this task can learn
a simple phase-coded oscillator solution
this solution should also be available, at least in the case of OUR non-delay, non-cosine
pulse cycling task. The point of this task is thus to find out something about
the conditions that allow the phase-coding solution in the SET case, and perhaps
find out why this kind of solution is not available
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class Setcard(Dataset):

	def __init__(self, task_params):
		self.task_params = task_params

	def __len__(self):
		return self.task_params["num_trials"]

	def __getitem__(self, idx):

		trial, label = self.generate_stim(idx)
		target, mask = self.generate_target(idx, label, trial)

		return trial, target, mask

	def generate_stim(self, idx):

		#Last third of trials is = 0 mod 3
		if idx >= 2*self.task_params["num_trials"]/3:
			label = 0
		#Second third of trials is = 1 mod 3
		elif idx >= self.task_params["num_trials"]/3:
			label = 1
		#this is just if you wanted to do multi-class labels
		elif idx < self.task_params["num_trials"]/3:
			label = 2

		final_index = -10
		T = self.task_params["trial_len"]
		trial = torch.zeros(T).unsqueeze(dim=1)

		#select a subset of input presentation times => should be small
		ptimes = np.random.choice([0,1], 
			np.ceil(self.task_params["trial_len"]/3), 
			p=[.9, .1]
			)
		
		#generate in this way, rather than randomnly to ensure dataset balance
		for i in range(T+final_index):

			if ptimes[i]==1:
				level = np.random.choice([0,1,2])
				trial[3*i:3*i+3]=np.array([level, level, level])
				
			if i == T+final_index-1:

				if label == 0:
					trial[i:i+3]=(3-(3-trial.sum())%3)%3
					
				elif label==1:
					trial[i:i+3]=(1-(3-trial.sum())%3)%3

				elif label==2:
					trial[i:i+3]=(2-(3-trial.sum())%3)%3

		label = 1 if label==0 else 0
		
		return trial, label

	def generate_target(self, idx, label, trial):

		target = torch.zeros(self.task_params["trial_len"], dtype=int)
		target[-15:-1]=1 if trial.sum()%3==0 else 0
		mask = torch.zeros_like(target, dtype=bool)
		mask[-15:-1]=1

		return target, mask