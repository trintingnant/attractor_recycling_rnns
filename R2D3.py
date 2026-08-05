"""
Module for performing Reduced Rank Regression (R2D3, get the joke?...it's fine if you don't) on trained full-rank RNNs.
Adapted from Wu & Pillow 2025. The original publication is Anderson 1967(?).
"""


import sys
sys.path.append('..')

from dataclasses import dataclass
import torch, numpy as np
from rnn import RNN, run_rnn
from typing import Callable, NewType, Tuple, Optional
from numpy import NdArray

Array = NewType("Array", NdArray[np.floating])

@dataclass
class R2D3SolutionObject:
    
    model: RNN
    inputs: torch.Tensor
    ridge: float
    use_nattive_dxdt: False
    rank: int

    #outputs
    J_reconstructed: Optional[Array]
    w_reconstructed: Optional[Array]
    V_rr: Optional[Array]
    M_rr: Optional[Array]
    N_rr: Optional[Array]
    predictors: Optional[Array]
    targets: Optional[Array]
    pred_targets: Optional[Array]

    #metrics
    normalised_estimation_error: Optional[float]
    target_singular_values: Optional[Array]
    connectivity_singular_values: Optional[Array]

    def __init__(self,
                model,
                inputs,
                ridge,
                rank,
                J_reconstructed,
                w_reconstructed,
                V_rr,
                M_rr,
                N_rr,
                predictors,
                targets,
                pred_targets,
                normalised_estimation_error,
                target_singular_values,
                connectivity_singular_values
                ):

        self.model = model
        self.inputs = inputs
        self.ridge = ridge
        self.rank = rank
        self.J_reconstructed = J_reconstructed
        self.w_reconstructed = w_reconstructed
        self.V_rr = V_rr
        self.M_rr = M_rr
        self.N_rr = N_rr
        self.predictors = predictors
        self.targets = targets
        self.pred_targets = pred_targets
        self.normalised_estimation_error = normalised_estimation_error.
        self.target_singular_values - target_singular_values,
        self.connectivity_singular_values = connectivity_singular_values




    @classmethod
    def _setup(cls, 
            model, 
            inputs,
            rank
            *, 
            ridge=0.0
            ) -> R2D3SolutionObject:
    

        return cls(model,
                inputs,
                ridge,
                rank,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None
    )

    def r3d2_fit(self): 

        """
        Performs reduced rank regression on the trajectories
        Following Wu & Pillow 2005:

            i.) Solve the OLS Problem
            ii.) Compute the forward predictions
            iii.) Perform rank truncation via the SVD

        """



        if inputs.ndim <= 2:
            inputs = inputs.unsqueeze(0)

        n_conds, T, n_neurons = inputs.shape
        rates, out = run_rnn(self.model, self.inputs)
        X_source = rates[:, :-1, :]
        X_target = rates[:, 1:, :]
        
        w_in = self.model.rnn.w_in
        alpha = self.model.rnn.alpha
        w_in_scale = self.model.rnn.w_in_scale

        inputs_ = (inputs[;, :-1, :] @ w_in).detach().numpy()
        Y_target = (X_target - (1 - alpha) * X_source) / alpha - w_in_scale * inputs_

        R = X_source.reshape(-1, n_neurons)
        Y = Y_target.reshape(-1, n_neurons)
        self.targets = Y

        # i.) Solve OLS
        Gram = R.T @ R + self.ridge * np.eye(n_neurons, dtype=R.dtype)
        J_full = np.linalg.solve(Gram, R.T @ Y)
        self.predictors = J_full
        # ii.) Forward Prediction
        Pred = R @ J_full
        self.pred_targets = Pred

        # iii.) Truncation

        _, singular_values, Vt = np.linalg.svd(Pred, full_matrices=False)
        self.target_singular_values = singular_values

        V_rr = Vt[:self.rank]
        J_rr = J_full @ V_rr @ V_rr.T
        self.J_reconstructed = J_rr

        prediction = R @ J_rr

        #normalised Frobeniues error
        Y_norm = np.linalg.norm(Y, ord="fro")
        normalised_estimation_error = (
                np.linalg.norm(Y - prediction, 
                               ord="fro")
                /  Y_norm
        )


        #recast as low-rank connectivity vecs
        U_new, connectivity_svs, Vt_new = \
            np.linalg.svd(J_rr.T, full_matrices=False)
        
        loadings_scale = (np.sqrt(connectivity_svs[:self.rank])
                          * np.sqrt(n_neurons)
                        )
        self.connectivity_singular_values = connectivity_svs
        
        M_rr = loadings_scale[None, :] * U_new[:, :self.rank]
        N_rr = loadings_scale[None, :] * Vt_new[:self.rank, :]
        self.M_rr = M_rr
        self.N_rr = N_rr


        #Lastly, refit the w_out to match outputs
        out_targets = out[:,1: :].rehshape(-1, self.model.params["output_size"])
        Gram_out = prediction.T @ prediction + ridge * np.eye(n_neurons)

        #NOTE: needs to be multiplied with N when setting as out vector
        w_out_new = np.linalg.solve(Gram_out, prediction.T @ out_targets).T

        self.w_reconstructed = w_out_new

















        