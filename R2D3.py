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
from numpy.typing import NDArray

Array = NewType("Array", NDArray[np.floating])

@dataclass
class R2D3SolutionObject:
    
    model: RNN
    inputs: torch.Tensor
    ridge: float
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
    predictions: Optional[Array]
    target_singular_values: Optional[Array]
    connectivity_singular_values: Optional[Array]

    #metrics
    normalised_estimation_error: Optional[float]
    

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
                predictions,
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
        self.predictions = predictions
        self.normalised_estimation_error = normalised_estimation_error
        self.target_singular_values = target_singular_values
        self.connectivity_singular_values = connectivity_singular_values




    @classmethod
    def _setup(cls, 
            model, 
            inputs,
            rank,
            ridge=0.0
            ):
    

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



        if self.inputs.ndim <= 2:
            self.inputs.unsqueeze(0)

        rates, out = run_rnn(self.model, self.inputs)
        n_conds, T, n_neurons = rates.shape
        X_source = rates[:, :-1, :]
        X_target = rates[:, 1:, :] 
        #actitvations 0-> rates
        r_source = np.tanh(X_source)

        w_in = self.model.rnn.w_in
        alpha = self.model.rnn.alpha
        w_in_scale = self.model.rnn.w_in_scale.item()

        inputs_ = (self.inputs[:, :-1, :] @ w_in).detach().numpy()
        
    
        Y_target = (X_target - (1 - alpha) * X_source) / alpha - inputs_

        R = r_source.reshape(-1, n_neurons)
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
        
        V_rr = Vt[:self.rank].T
        J_rr = J_full @ V_rr @ V_rr.T
        self.J_reconstructed = J_rr

        prediction = R @ J_rr
        self.predictions = prediction.reshape(n_conds, T - 1, n_neurons)

        #normalised Frobeniues error
        Y_norm = np.linalg.norm(Y, ord="fro")
        normalised_estimation_error = (
                np.linalg.norm(Y - prediction, 
                               ord="fro")
                /  Y_norm
        )

        self.normalised_estimation_error = normalised_estimation_error


        #recast as low-rank connectivity vecs
        U_new, connectivity_svs, Vt_new = \
            np.linalg.svd(J_rr.T, full_matrices=False)
        
        loadings_scale = (np.sqrt(connectivity_svs[:self.rank])
                          * np.sqrt(n_neurons)
                        )
        self.connectivity_singular_values = connectivity_svs
        
        M_rr = loadings_scale[None, :] * U_new[:, :self.rank]
        N_rr = loadings_scale[:, None] * Vt_new[:self.rank, :]
        self.M_rr = M_rr
        self.N_rr = N_rr


        #Lastly, refit the w_out to match outputs
        out_targets = out[:,1: :].reshape(-1, self.model.params["output_size"])
        Gram_out = prediction.T @ prediction + self.ridge * np.eye(n_neurons)

        #NOTE: needs to be multiplied with N when setting as out vector

        w_out_new = np.linalg.solve(Gram_out, prediction.T @ out_targets).T

        self.w_reconstructed = w_out_new


    def build_LowRankRNN(self) -> RNN:

        """
        constructs the low-rank RNN from fits 
        """

        params = self.model.params.copy()
        params["low_rank"] = True
        params["rank"] = self.rank
        w_in = self.model.rnn.w_in  
        w_out = self.w_reconstructed
        model = RNN(params)

        with torch.no_grad():
            model.rnn.m.set_(torch.from_numpy(M))
            model.rnn.n.set_(torch.from_numpy(N))
            model.rnn.w_in.copy_(w_in)
            model.rnn.w_out.set_(torch.from_numpy(params["recurrent_size"] * w_out.astype(np.float32)).T)

        return model














        