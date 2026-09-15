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
        Y_target = (X_target - (1 - alpha) * X_source) / alpha - w_in_scale * inputs_

        R = r_source.reshape(-1, n_neurons)
        Y = Y_target.reshape(-1, n_neurons)
        self.targets = Y

        # i.) Solve OLS
        Gram = R.T @ R + self.ridge * np.eye(n_neurons, dtype=R.dtype)
        J_full = np.linalg.solve(Gram, R.T @ Y)
        self.predictors = J_full # These are NOT the predictors. That's just the coefficient matrix. 
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

        #normalised Frobeniues error
        Y_norm = np.linalg.norm(Y, ord="fro")
        normalised_estimation_error = (
                np.linalg.norm(Y - prediction, 
                               ord="fro")
                /  Y_norm
        )

        self.normalised_estimation_error = normalised_estimation_error

        # final hidden state prediction
        prediction_r = prediction.reshape(n_conds, T - 1, n_neurons)
        h_prediction = (
                (1 - alpha) * X_source
                + alpha * (
                prediction_r + w_in_scale * inputs_
            )
        )

        self.prediction = h_prediction

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

        W_from_loadings =  (
                M_rr @ N_rr
                ) / n_neurons

        factorisation_error = (
            np.linalg.norm(
                J_rr.T - W_from_loadings
                ord="fro"
                )
            /
            np.linalg.norm(
                J_rr.T
                ord="fro"
                )
            )
        self.factorisation_error = factorisation_error

        # Lastly, refit the w_out to match outputs

        h_prediction = h_prediction.reshape(-1, n_neurons)
        Gram_out = h_prediction.T @ h_prediction + self.ridge * np.eye(n_neurons)

        out_targets = out[:,1: :].reshape(-1, self.model.params["output_size"])

        #NOTE: needs to be multiplied with N when setting as out vector
        w_out_new = np.linalg.solve(Gram_out, h_prediction.T @ out_targets).T

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
            model.rnn.m.set_(torch.from_numpy(M))
            model.rnn.n.set_(torch.from_numpy(N))
            params["w_in_scale"] * model.rnn.w_in.copy_(w_in)
            model.rnn.w_out.set_(torch.from_numpy(params["recurrent_size"] * w_out.astype(np.float32)).T)

        return model

def jr3_fit(self):

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

    for t, gain in enumerate(gains):

        D = np.diag(gain)

        # these are the true jacobians
        J_t = -I + D @ K_true

        # these are our regression targets
        Y_t = J_t + I 

        # we use the "accumulation trick" instead of stacking here to conserve memory
        Gram_data += D.T @ D
        R += D.T @ Y_t

    #We solve Y_t = D @ W.T 
    Gram = Gram_data + self.ridge * I
    K_full = np.linalg.solve(Gram, R)
    

    prediction_covariance = K_full.T @ Gram_data @ K_full

    eigvals, eigvecs = np.linalg.eigh(prediction_covariance)
    sorted = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted]
    eigvecs = eigvecs[:, sorted]

    # shouldn't happen, but for sanity set all negative eigvals to 0.0
    eigvecs = np.maximum(eigvecs, 0.0)
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
    target_norm = np.trace(K_true @ Gram_data @ K_true)

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
            K_rr.T - W_from_loadings
            ord="fro"
        )
        /
        np.linalg.norm(
            K_rr.T
            ord="fro"
        )
    )

    # unfortunate attribute naming now:
    self.J_reconstructed = K_rr.T
    self.normalised_estimation_error = normalised_jac_pred_error
    self.factorisation_error = factorisation_error
    


  









        