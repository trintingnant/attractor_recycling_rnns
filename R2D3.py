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
from tqdm.auto import tqdm
from utils import mse

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
                rank,                   # this is what I call the 'reconstruction rank'
                mechanistic_rank,       # this is what I call the 'mechanistic rank' 
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
                connectivity_singular_values,
                factorisation_error
                ):

        self.model = model
        self.inputs = inputs
        self.ridge = ridge
        self.rank = rank
        self.mechanistic_rank = mechanistic_rank
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
        self.factorisation_error = factorisation_error

    @classmethod
    def _setup(cls, 
            model, 
            inputs,
            rank,
            mechanistic_rank=None,
            ridge=0.0
            ):
        
        if mechanistic_rank is None or mechanistic_rank > rank:
            mech_rank = rank

        else:
            mech_rank = mechanistic_rank

        return cls(model,
                inputs,
                ridge,
                rank,
                mech_rank, 
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
                None,
                None
    )

    def r2d3_tube_fit(
                self,
                perturbation_scales=(0.005, 0.01, 0.02),
                n_perturbations=2,
                seed=None,
                ):

        if self.inputs.ndim <= 2:
            self.inputs = self.inputs.unsqueeze(0)

        rates, out = run_rnn(self.model, self.inputs)

        n_conds, T, n_neurons = rates.shape

        X_source = rates[:, :-1, :]
        X_target = rates[:, 1:, :]

        r_source = np.tanh(X_source)

        w_in = self.model.rnn.w_in
        alpha = self.model.rnn.alpha
        w_in_scale = self.model.rnn.w_in_scale.item()

        inputs_ = (
            (self.inputs[:, :-1, :] @ w_in)
            .detach()
            .numpy()
        )

        Y_target = (
            (X_target - (1 - alpha) * X_source) / alpha
            - w_in_scale * inputs_
        )

        R_original = r_source.reshape(-1, n_neurons)

        Y_original = Y_target.reshape(-1, n_neurons)

        W_full = (
            self.model.rnn.w_rec
            .detach()
            .numpy()
        )

        K_true = W_full.T

        state_std = X_source.std(axis=(0, 1), keepdims=True)

        state_std = np.maximum(state_std, 1e-6)

        rng = np.random.default_rng(seed)

        R_tube = []
        Y_tube = []

        for scale in perturbation_scales:

            for _ in range(n_perturbations):

                noise = (
                    scale
                    * state_std
                    * rng.standard_normal(
                        X_source.shape
                    )
                )

                X_perturbed = X_source + noise
                r_perturbed = np.tanh(X_perturbed)
                R_perturbed = r_perturbed.reshape(-1, n_neurons)
                Y_perturbed = (R_perturbed @ K_true)

                R_tube.append(R_perturbed)

                Y_tube.append(Y_perturbed)

        if len(R_tube) > 0:

            R = np.concatenate(
                [R_original] + R_tube,
                axis=0
            )

            Y = np.concatenate(
                [Y_original] + Y_tube,
                axis=0
            )

        else:

            R = R_original
            Y = Y_original

        self.targets = Y

        Gram = (
            R.T @ R
            + self.ridge
            * np.eye(
                n_neurons,
                dtype=R.dtype
            )
        )

        J_full = np.linalg.solve(Gram, R.T @ Y)

        self.predictors = J_full

        Pred = R @ J_full

        self.pred_targets = Pred

        # this is the PCA step, singular values of Pred are eigenvalues of Pred^T Pred
        _, singular_values, Vt = np.linalg.svd(Pred, full_matrices=False)

        self.target_singular_values = singular_values
        
        V_rr = Vt[:self.rank].T
        self.V_rr = V_rr

        J_rr = J_full @ V_rr @ V_rr.T
        self.J_reconstructed = J_rr

        prediction = R @ J_rr
    
        self.normalised_estimation_error = (
            np.linalg.norm(
                Y - prediction,
                ord="fro"
            )
            /
            np.linalg.norm(
                Y,
                ord="fro"
            )
        )

        prediction_original = R_original @ J_rr
        prediction_r = prediction_original.reshape(n_conds, T - 1, n_neurons)

        h_prediction = (
            (1 - alpha) * X_source
            + alpha * (
                prediction_r
                + w_in_scale * inputs_
            )
        )

        self.prediction = h_prediction

        U_new, connectivity_svs, Vt_new = np.linalg.svd(J_rr.T, full_matrices=False)

   

        loadings_scale = (
            np.sqrt(
                connectivity_svs[:self.mechanistic_rank]
            )
            * np.sqrt(
                n_neurons
            )
        )

        M_rr = (
            loadings_scale[None, :]
            * U_new[:, :self.mechanistic_rank]
        )

        N_rr = (
            loadings_scale[:, None]
            * Vt_new[:self.mechanistic_rank, :]
        )

        self.M_rr = M_rr
        self.N_rr = N_rr

        self.connectivity_singular_values = connectivity_svs

        W_from_loadings =  M_rr @ N_rr / n_neurons

        self.factorisation_error = (
            np.linalg.norm(
                J_rr.T - W_from_loadings,
                ord="fro"
            )
            /
            np.linalg.norm(
                J_rr.T,
                ord="fro"
            )
        )

        h_prediction_flat = h_prediction.reshape(-1, n_neurons)

        Gram_out = (
            h_prediction_flat.T
            @ h_prediction_flat
            + self.ridge
            * np.eye(n_neurons)
        )

        out_targets = out[:, 1:, :].reshape(-1, self.model.params["output_size"])

        w_out_new = np.linalg.solve(
            Gram_out,
            h_prediction_flat.T @ out_targets
        ).T

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
            M = self.M_rr
            N = self.N_rr

            with torch.no_grad():
                model.rnn.m.set_(torch.from_numpy(M.astype(np.float32)))
                model.rnn.n.set_(torch.from_numpy(N.astype(np.float32)))
                model.rnn.w_in.copy_(w_in.to(dtype=model.rnn.w_in.dtype))
                model.rnn.w_out.set_(torch.from_numpy(
                                    params["recurrent_size"] 
                                    * w_out.astype(np.float32)).T
                                    )

            return model
    
    @staticmethod
    def run4model(model, inputs, tol=.05):

        rates, out = run_rnn(model, inputs)
        errors = []
        solutions = []

        for rank in tqdm(range(1, args.r), unit="rank"):
            Solution = R2D3SolutionObject._setup(model, inputs, rank=rank, ridge=1e-1, mechanistic_rank=None)
            Solution.r2d3_tube_fit(n_perturbations=4)
            solutions.append(Solution)
            errors.append(Solution.normalised_estimation_error)

        # we still wa
        for idx in range(len(solutions)):
            sol = solutions[idx]
            lr_model = sol.build_LowRankRNN()
            _, out = run_rnn(lr_model, trials)
            err = mse(out, targets, masks)
            if err <= tol:
                break

        # build the LR model
        lr_model = sol.build_LowRankRNN()

        return sol, errors, lr_model







        