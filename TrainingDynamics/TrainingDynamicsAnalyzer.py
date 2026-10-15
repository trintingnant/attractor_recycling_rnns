from functools import cached_property
from typing import Callable, Dict, Any
from train import TrainingLogger
from utils import to_numpy

import torch, numpy as np

class DynamicsAnalyzer: 

    """
    Lazy container for carrying out analyses post training:

        -- Common quantities are cached properties, computed once at outset
        and pushed to the cache (lazy evaluation)
        -- Can add and handle arbitrary callables and execute on the stuff in logger
       

    """

    def __init__(self, logger):

        self.logger = logger
        self._callables: Dict[str, Callable] = {}
        self._results: Dict[str, Any] = {}

    @staticmethod 
    def to_numpy(x):
        return to_numpy(x)
    
    # We start by caching some properties we usually need. These can be removed
    # in favour of dynamical property calls via 'add' for performance. 
    # but it's just nicer not to have to manually define everything every time you 
    # create an Analyzer object

    @cached_property
    def epochs(self):
        
        return np.asarray(
            self.logger.epochs
        )

    @cached_property
    def params(self):
        return self.logger["params"]
    
    @cached_property
    def gradients(self):
        return self.logger["grads"]

    @cached_property
    def W_in(self):

        return np.stack([
            self.to_numpy(
                param["w_in"]
            )
            for param in self.params
        ])
    
    @cached_property
    def W_rec(self):

        return np.stack([
            self.to_numpy(
                param["w_rec"]
            )
            for param in self.params
        ])
    
    @cached_property
    def W_out(self):

        return np.stack([
            self.to_numpy(
                param["w_out"]
            )
            for param in self.params
        ])


    @cached_property
    def dW_in(self):
        return self.W_in - self.W_in[0]
    
    @cached_property
    def dW_rec(self):
        return self.W_rec - self.W_rec[0]

    @cached_property
    def dW_out(self):
        return self.W_out - self.W_out[0]

    @cached_property
    def rec_svd(self):

        Us = []
        Ss = []
        Vs = []

        for dW in self.dW_rec:
             
            U, S, Vt = np.linalg.svd(
                dW,
                full_matrices=False
            )

            Us.append(U)
            Ss.append(S)
            Vs.append(Vt.T)

        return {
            "U": np.stack(Us),
            "S": np.stack(Ss),
            "V": np.stack(Vs)
        }
        
    @cached_property
    def rec_singular_values(self):
        return self.rec_svd["S"]
    
    @cached_property 
    def rec_U(self):
        return self.rec_svd["U"]
    
    @cached_property
    def rec_V(self):
        return self.rec_svd["V"]
    


    def __getitem__(self, name):

        if name not in self._results:

            try:
                self.execute(name)

            except Exception as e:
                print(e)
                 
        return self._results[name]

    #self registry: allow adding arbitrary functions to _calllables
    def add(
            self,
            name: str,
            fn: Callable,
            overwrite: bool = False
        ):

        if name in self._callables and not overwrite:
            raise KeyError(
                f"{name} already registered and can't be overwritten"
                )
        
        self._callables[name] = fn

        #In case of overwrite, flush old results from _results
        self._results.pop(name, None)

        return self
    

    def remove(self, name):

        self._callables.pop(name, None)
        self._results.pop(name, None)

        return self
    
    def execute(self, name, force=False):

        """
        runs a given callable, overwriting previous results only if force is True
        """

        if name in self._results and not force:
            return self._results[name]
        
        if name not in self._callables:
            raise KeyError(
                f"{name} not a registered function"
            )
        
        result =  self._callables[name](self)
        self._results[name] = result

        return result


    
    

        
    


             
    




