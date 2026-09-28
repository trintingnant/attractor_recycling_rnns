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

    def jr3_fit(self, subspace_dim):

        from utils import phiPrime

        """
        This performs reduced rank regression ON THE JACOBIANS
        Procedure as above, but the regression targets are the 
        jacobians along the trajectory.

        This can be considered a dual of the EMPJ method (Pollock & Jazayeri, 2020)

        Why is this an improvement? The standard reduced regression
        uses the trajectories as pointwise regression targets -> small errors 
        can blow up quickly during the forward pass of the reconstructed networks. 
        """
        
        if self.inputs.ndim <= 2:
                self.inputs = self.inputs.unsqueeze(0)

        rates, out = run_rnn(self.model, self.inputs)
        n_conds, T, n_neurons = rates.shape

        W_full = (
            self.model.rnn.w_rec
            .detach()
            .numpy() 
        ) 

        # regression target 
        K_true = W_full.T 

        # compute the gains along the trajectory 
        rates_flat = rates.reshape(-1, n_neurons)
        gains = phiPrime(rates_flat)
        n_samples = gains.shape[0] 
        # how many points we're using. This is analogous to the setpoints in EMPJ
        # We could sample the activations more coarsely
        
        # obtain the Jacobians: J(t) = -I * PhiPrime(h(t)) W_rec.T/K_true
        # ATTENTION: DO NOT STORE THE JACOBIANS

        I = np.eye(n_neurons, dtype=K_true.dtype)
        Gram_data = np.zeros((n_neurons, n_neurons), dtype=K_true.dtype)
        R = np.zeros_like(Gram_data)
        
        # we can't do this on the full Jacobians. Like in EMPJ we want to fit only 
        # the leading directions of the Jacobian

        rates_centered = (
            rates_flat
            - rates_flat.mean(
                axis=0,
                keepdims=True,
            )
        )

        _, trajectory_svs, trajectory_Vt = np.linalg.svd(
            rates_centered,
            full_matrices=False,
        )

        Q = trajectory_Vt[:subspace_dim].T


        for t, gain in enumerate(gains):

            R_t = Q.T * gain[None, :]
        

            Y_t = R_t @ K_true

            # we use the "accumulation trick" instead of stacking here to conserve memory
            Gram_data += R_t.T @ R_t
     
            R += R_t.T @ Y_t
        


        #We solve Y_t = D @ W.T 
        Gram = Gram_data + self.ridge * I
        K_full = np.linalg.solve(Gram, R)
        

        prediction_covariance = K_full.T @ Gram_data @ K_full

        eigvals, eigvecs = np.linalg.eigh(prediction_covariance)
        sorted = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted]
        eigvecs = eigvecs[:, sorted]

        # shouldn't happen, but for sanity set all negative eigvals to 0.0
        eigvals = np.maximum(eigvals, 0.0)
        # singular value of the target matrix are the roots of the eigenvectors of the covariance
        target_svs = np.sqrt(eigvals)

        V_rr = eigvecs[:, :self.rank]
        self.V_rr = V_rr

        # this is the reduced rank regression step. Just as before. 
        K_rr = K_full @ V_rr @ V_rr.T

        #errors 
        K_error = K_true - K_rr

        # We want || Y_t - Y_est ||
        # Since Y_t = D @ K_true, Y_est = D @ K_rr
        # if we comp || D(t)K_true - D(t)K_rr ||^2_F, sum over the t and take the sqrt, that should work. 
        # also: || D(t)K_true - D(t)K_rr ||^2_F = tr(K_error.T Sum D.T D K_error)
        # middle term is just Gram_data: NEAT!

        rec_error_norm = np.trace(K_error.T @ Gram_data @ K_error)
        target_norm = np.trace(K_true.T @ Gram_data @ K_true)

        normalised_jac_pred_error = np.sqrt(rec_error_norm) / np.sqrt(target_norm)

        # as before: extract low-rank network loadings, This is just verbatim the same code block.

        U_new, connectivity_svs, Vt_new = \
                np.linalg.svd(K_rr.T, full_matrices=False)
            
        loadings_scale = (np.sqrt(connectivity_svs[:self.rank])
                        * np.sqrt(n_neurons)
                        )
        self.connectivity_singular_values = connectivity_svs
        
        M_rr = loadings_scale[None, :] * U_new[:, :self.rank]
        N_rr = loadings_scale[:, None] * Vt_new[:self.rank, :]
        self.M_rr = M_rr
        self.N_rr = N_rr
        
        W_from_loadings =  (
            M_rr @ N_rr
        ) / n_neurons

        factorisation_error = (
            np.linalg.norm(
                K_rr.T - W_from_loadings,
                ord="fro"
            )
            /
            np.linalg.norm(
                K_rr.T,
                ord="fro"
            )
        )

        # unfortunate attribute naming now:
        self.J_reconstructed = K_rr.T
        self.normalised_estimation_error = normalised_jac_pred_error
        self.factorisation_error = factorisation_error
        self.w_reconstructed = (self.model.rnn.w_out
                                .detach()
                                .numpy() 
                                ).T 







        