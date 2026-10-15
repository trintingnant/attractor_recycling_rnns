from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Optional, Iterable
import torch, numpy as np


class Task(Dataset, ABC):

    def __init__(self, task_params):
        self.task_params = task_params

    def __len__(self):
        return self.task_params["num_trials"]
    
    def __getitem__(self, idx):
        trial, _, _ = self.get_stim(idx)
        target, mask = self.get_target(idx)
        return trial, target, mask
    
    @classmethod
    def __subclasshook__(cls, otherCls):
        if isinstance(cls, Task):
            #any subclass must implement Dataset dunders
            if all([
                any("__len__" in other.__dict__ for other in otherCls.__mro__),
                any("__getitem__" in other.__dict__ for other in otherCls.__mro__)]
            ):
                return True
            
        else:
            raise NotImplemented
        
    @classmethod
    def _subclass(cls, name):
        for cls in cls.__subclasses__():
            if cls.__name__() == name:
                return cls
            else:
                raise ValueError("no subclass found")
            
    @staticmethod
    @abstractmethod
    def __name__():
        """convenience method; allows passing tasks to TaskManager using strings"""
        raise NotImplementedError
    
    @abstractmethod
    def get_stim(self, idx: int):
        """stimulus generator: to be implemented in subclass"""
        raise NotImplementedError

    @abstractmethod
    def get_target(self, idx: int):
        """target generator: to be implemented in subclass"""
        raise NotImplementedError

    @staticmethod
    def load_task(name: str, task_params):
        #search through subclasses the correct task
        for cls in Task.__subclasses__():
            if cls.__name__ == name:
                return cls(task_params)

        raise ValueError(f"Task {name} not yet implemented")

    @abstractmethod
    def generate_inputs(self, num_inputs):
        """generate interpolated inputs for a task"""
        raise NotImplementedError


def make_task_from_tensors(trials, targets, masks, name="") -> Task:

    """
    Converts a set of trials, targets and masks to a local Task object.

    @Args:

    trials, targets, masks: torch.Tensors

    NOTE: tensors should be detached when passed, otherwise the backward pass double-dips
    the computational graph

    @Returns:

    TensorTask: a local Task subclass
    """
    if len(trials.shape) <= 2:
        trials = trials.unsqueeze(dim=0)
    if len(targets.shape) <= 2:
        targets = targets.unsqueeze(dim=0)
    if len(masks.shape) <= 2:
        masks = masks.unsqueeze(dim=0)

    batch_size, _, _ = trials.shape
    task_params = {"num_trials": batch_size} #basically a  dummy variable

    class TensorTask(Task):

       def __name__():
           return name

       def get_stim(self, idx):
           #labels, delays set by trial tensors
           return trials[idx], 0, 0

       def get_target(self, idx):
           mask = torch.ones_like(targets[idx], dtype=bool)
           return targets[idx], mask
       
       def generate_inputs(self, num_inputs):
           pass
       
    return TensorTask(task_params)



def combine_tasks(tasks) -> Task:

    """Return instance of CombinedTask, where CombinedTask is a generic class
    the instances of which depend the tasks that are passed as arguments"""

    from bisect import bisect
    num_tasks = len(tasks)

    def merge_task_params(tasks):

        """
        merges the relevant params for each task into a single dictionary for DataLoader handling
        """
        mtask_params = {}
        mtask_params.update({"num_trials": np.sum([val.get("num_trials") for val in tasks.values()])})
        #assumes all tasks have the same in_dim (required)
        mtask_params.update({"in_dim": len(tasks)+tasks[list(tasks.keys())[0]]["in_dim"]})
        #assumes all tasks have the same trial_len (required)
        mtask_params.update({"trial_len": tasks[list(tasks.keys())[0]]["trial_len"]})
        #assumes all tasks have the same out_dim (required)
        mtask_params.update({"out_dim": tasks[list(tasks.keys())[0]]["out_dim"]})


        return mtask_params


    def build_idxs(idx):
        
        """builds the proper indeces for each trial and each task"""
        
        #get the indexes correspoding to each task
        task_idxs = np.cumsum([val.get("num_trials") for val in tasks.values()])
        #find correct task per index
        task_idx = bisect(task_idxs, idx)
        #define shift term to correct for number of trials per task
        shift = 0 if (task_idx == 0) else task_idxs[task_idx-1]

        return task_idxs, task_idx, shift

    class CombinedTask(Task):
        #ATTENTION: this (unfortunately) generates myriad subclasses of the abstract interface,
        #fbut fine...
        #need to manually garbage collect after each call -> TODO
        
        def __name__():
            return "CombinedTask"

        def get_stim(self, idx):

            task_idxs, task_idx, shift = build_idxs(idx)
            key, val = list(tasks.items())[task_idx]
            task = Task.load_task(key, val)
            #define shift term to correct for number of trials per task
            trial, label, delay = task.get_stim(idx-shift)
            #add control inputs
            trial = torch.cat((trial, torch.zeros(trial.shape[0], num_tasks)), dim=1)
            trial[delay:, val.get("in_dim")+task_idx] = torch.ones(val.get("trial_len")-delay)
            
            return trial, label, delay

        def get_target(self, idx):

            task_idxs, task_idx, shift = build_idxs(idx)
            key, val = list(tasks.items())[task_idx]
            task = Task.load_task(key, val)
            _, label, delay = task.get_stim(idx-shift)
            target, mask = task.get_target(idx-shift)

            return target, mask

        def generate_inputs(self, num_inputs):
            pass

    return CombinedTask(merge_task_params(tasks))





       








    

    


        