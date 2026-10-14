"""helper functions for training dynamics analysis"""

import torch, numpy as np # type: ignore
from TrainingDynamics.TrainingDynamicsAnalyzer import DynamicsAnalyzer
from scipy.signal import find_peaks, peak_widths # type: ignore
from copy import deepcopy

EPS = 1e-12
global MAX_RANK
MAX_RANK = 7

def smooth(x, window):

        """
        simple smooth window function
        """

        x = np.asarray(x, dtype=float)

        if window <= 1:
            return x.copy()

        left = window // 2
        right = window - 1 - left

        xp = np.pad(
            x, 
            (left, right),
            mode="edge"
        )

        # simple mean-kernel
        kernel = np.ones(window) / window

        return np.convolve(
            xp,
            kernel,
            mode="valid"
        )

# learning window detection for stagewise learning
def detect_learning_windows(A: DynamicsAnalyzer,
                            num_svs=7,
                            smooth_window=8,
                            min_distance=10,
                            prominence=.2,
                            rel_height=.1
    ): 

    """
    chops up the learning period into learning windows marked
    by learning events. Performs simple peak detection on the
    combined loss and weight changes.
    """

    losses = A["full_task_gradients_and_losses"]["losses"]
    epochs = A.epochs

    log_losses = np.log(losses) + EPS
    loss_speed = (
       -np.gradient(
            smooth(
               log_losses,
               smooth_window
            ),
            A.epochs
       )
    )
    
    # NOTE: cached_properties are passed by reference
    S = deepcopy(
        A.rec_singular_values[:, :num_svs]
        )
    
    S_norm = np.max(S, axis=0)[None, :] + EPS
    S /= S_norm

    structure_speed = np.linalg.norm(
        np.gradient(
            S,
            A.epochs,
            axis=0
        ),
        axis=1
    ) 
    structure_speed = smooth(structure_speed, smooth_window)

    # event score is computed over both the loss speed and the structure speed
    # normalise to percentile values, so the two quantities live on the same scale

    event_score = (
        loss_speed
        /
        (
            np.percentile(
                loss_speed,
                95
            )
            + EPS
        )

        + 

        structure_speed
        /
        (
            np.percentile(
                structure_speed, 
                95
            )
            + EPS
        )
    )

    event_score = smooth(
        event_score,
        smooth_window
    )

    peaks, properties = find_peaks(
        event_score,
        prominence=prominence,
        distance=min_distance
    )

    widths, _, left_ips, right_ips = peak_widths(
        event_score,
        peaks,
        rel_height=rel_height
    )

    events = []
    for k, peak in enumerate(peaks):

        start_idx = int(np.floor(left_ips[k]))
        end_idx = int(np.ceil(right_ips[k]))

        start_idx = max(0,start_idx)
        end_idx = min(len(epochs) - 1, end_idx)

        events.append({
            "center_idx": int(peak),
            "center_epoch": epochs[peak],
            "start_idx": start_idx,
            "start_epoch": epochs[start_idx],
            "end_idx": end_idx,
            "end_epoch": epochs[end_idx],
            "score": event_score[peak],
            "prominence": properties["prominences"][k],
            "width": widths[k],
        }
        )

    return {
        "events": events,
        "event_score": event_score,
        "loss_speed": loss_speed,
        "structure_speed": structure_speed,
        }

# computes the (marginal/cumulative) energy in the leading modes
# of the weight change matrix 

def recurrent_energy_spectrum(G, 
                              U, 
                              V, 
                              max_rank=MAX_RANK
    ):

    """
    Tangent-space decomposition of the gradients G in line with our 
    definition of the definition of the simplicity bias. Intended
    to illicit the 'gradient-trapping' phenomenon.

    The 're-used' portion is the part of the gradient that lies in the
    tangent space of the existing weight change at the start of the
    learning window, i.e.

        G_re = P_U G + G P_V - P_U G P_V

    We also take: 

        G_out = P_U G 
        G_in = G P_V

    These also have nice interpretations:
        
        G_out are the directions in which the gradient wants to modify the already existing output directions of the weight change
        G_in are the directions of the forward activity due to the weight change that are contributing to the gradient. In our lingo: this is the quantity
        that describes how much the existing structure is 'trapping' or 'hogging' the gradient.  
        
        
        the directions in which the forward activity due to the weight change is trapping the gradient

    So, hypothetically, gradient trapping should reveal itself in large G P_Vs for only a handful of right-singular vectors.
    At the same time, learning events should then be preceeded/accompanied by large excursions of P_U G. 

    In other words:
        large G P_V -> gradients lie primarily in the directions of the forward acticity due to already existing structure -> gradient trapping/hogging
        large P_U G -> gradients modify the weight change matrix in the output


    whereas the 'new' portion is simply the remaining part of the gradient 
    (the part that lies along the manifold normal). I.e.

        G_new = (I-P_U) G (I-P_V)

    """

    N = G.shape[0]
    I = np.eye(N)

    # We quantify the two parts as percentages of the total gradient norm
    # i.e. G_re + G_new = 1

    norm = (
        np.linalg.norm(
            G,
            "fro"
        ) ** 2
        + EPS
    )

    reuse = np.zeros(max_rank)
    new = np.zeros(max_rank)
    in_ = np.zeros(max_rank)
    out = np.zeros(max_rank)

    for r in range(1, max_rank + 1):

        Ur = U[:, :r]
        Vr = V[:, :r]

        PU = Ur @ Ur.T
        PV = Vr @ Vr.T

        G_in = G @ PV
        G_out = PU @ G
        Greuse = G_in + G_out - PU @ G @ PV
        Gnew = (I - PU) @ G @ (I - PV)


        # taking norms means all of these are cumulative
        in_[r - 1] = (
            np.linalg.norm(
                G_in,
                "fro"
            ) ** 2
            / norm
        )

        out[r - 1] = (
            np.linalg.norm(
                G_out,
                "fro"
            ) ** 2
            / norm
        )

        reuse[r - 1] = (
            np.linalg.norm(
                Greuse,
                "fro"
            ) ** 2
            / norm
        )

        new[r - 1] = (
            np.linalg.norm(
                Gnew,
                "fro"
            ) ** 2
            / norm
        )

    marginal = np.empty_like(reuse)
    marginal_in = np.empty_like(in_)
    marginal_out = np.empty_like(out)

    marginal[0] = reuse[0]
    marginal[1:] = reuse[1:] - reuse[:-1]

    marginal_in[0] = in_[0]
    marginal_in[1:] = in_[1:] - in_[:-1]

    marginal_out[0] = out[0]
    marginal_out[1:] = out[1:] - out[:-1]

    return (
            reuse, 
            new, 
            marginal, 
            marginal_in,
            marginal_out
        )

def vector_energy_spectrum(X, Q, max_rank):
      
    """Analogous to above, except x is a vector and Q
    is a nested set of directions, i.e.
    span{Q} = (Q1, Q2, ...)

    e.g. Q is the subspace spanned by the singular 
    directions of the weight change matrix truncated
    at different ranks
    """

    eps = 1e-12
    norm = np.linalg.norm(X) ** 2 + eps

    cumulative = np.zeros(max_rank)

    for r in range(1, max_rank + 1):

        Qr = Q[:, :r]
        proj = Qr @ (Qr.T @ X)
        cumulative[r-1] = np.linalg.norm(proj) ** 2 / norm
        
    marginal = np.empty_like(cumulative)
    marginal[0] = cumulative[0]
    marginal[1:] = cumulative[1:] - cumulative[:-1]

    return cumulative, marginal