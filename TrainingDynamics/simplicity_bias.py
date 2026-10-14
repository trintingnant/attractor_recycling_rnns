"""
Collection of functions we might want for testing the simplicity bias
"""

from TrainingDynamics.TrainingDynamicsAnalyzer import DynamicsAnalyzer
from TrainingDynamics.helpers import *
import torch, numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy
from utils import to_numpy


#gradient computation (in case grads not logged)
def compute_full_task_gradients_and_losses(
        params,
        model,
        task,
        loss_func,
        init_hidden=None
):
    
    loader = DataLoader(
        task,
        batch_size=len(task),
        shuffle=False
    )

    x, y, mask = next(iter(loader))

    checkpoint_model = deepcopy(model)

    losses = [] # losses
    G_rec = [] # recurrent gradient
    G_in_vecs = [] # input gradient
    G_out_vecs = [] # output gradient
    rates_ = [] # forward dynamix

    for checkpoint in params:

        checkpoint_model.rnn.load_state_dict(
            checkpoint,
            strict=False
        )

        checkpoint_model.zero_grad(
            set_to_none=True
        )

        checkpoint_model.eval()

        rates, y_pred = checkpoint_model(
            x,
            init_hidden
        )

        loss = loss_func(
            y_pred,
            y, 
            mask 
        )
        
        loss.backward()

        named_params = dict(
            checkpoint_model
            .rnn
            .named_parameters()
        )

        # recurrent gradient
        grec = (
            named_params["w_rec"]
            .grad
            .detach()
            .numpy()
            .copy()
        )

        # input gradient 
        gin = (
            named_params["w_in"]
            .grad
            .detach()
            .numpy()
            .copy()
        )

        # output gradient

        gout = (
            named_params["w_out"]
            .grad
            .detach()
            .numpy()
            .copy()
        )

        rates_.append(to_numpy(rates))
        losses.append(loss.item())
        G_rec.append(grec)
        G_in_vecs.append(gin)
        G_out_vecs.append(gout)


    return {
        "rates": np.asarray(rates_),
        "losses": np.asarray(losses),
        "G_rec": np.stack(G_rec),
        "G_in": np.stack(G_in_vecs),
        "G_out": np.stack(G_out_vecs),
    }


def compute_effective_ranks(analyzer: DynamicsAnalyzer):

    """
    Computes the effective rank of the recurrent gradient
    throughout training: should change depending on learning
    period
    """

    try:
        G_rec = analyzer["full_task_gradients_and_losses"]["G_rec"]

    except Exception as e:
        print(e)
    
    eff_ranks = []

    for idx in range(G_rec.shape[0]):
        _, S, _ = np.linalg.svd(
            G_rec[idx, ...]
        )

        eff_ranks.append(
            (S ** 2).sum() ** 2
            /
            (S ** 4).sum()
        )

    return np.stack(eff_ranks)



def compute_stagewise_gradient_energy(analyzer: DynamicsAnalyzer):

    try: 
        learning_events = detect_learning_windows(analyzer)

    except Exception as e:
        print(e)

    try:
        G_rec = analyzer["full_task_gradients_and_losses"]["G_rec"]

    except Exception as e:
        print(
            e.message,
            e.args
        )

    centres = np.asarray(
        [
        event["center_idx"] 
        for event in learning_events["events"]
        ]
    )

    U_rec_at_centers = analyzer.rec_U[centres]
    V_rec_at_centers = analyzer.rec_V[centres] 

    G_reuse = []
    G_new = []
    G_marginal = []
    G_marginal_in = []
    G_marginal_out = []


    for idx, (start, end) in enumerate(zip(centres[:-1], 
                                           centres[1:])
        ):
        

        U = U_rec_at_centers[idx]
        V = V_rec_at_centers[idx]
        G = G_rec[start:end, ...]
        
        for i in range(G.shape[0]):

            (re, 
            new, 
            marginal, 
            marginal_in, 
            marginal_out) = recurrent_energy_spectrum(
                                            G[i,...],
                                            U,
                                            V
                                ) 


            G_reuse.append(re)
            G_new.append(new)
            G_marginal.append(marginal)
            G_marginal_in.append(marginal_in)
            G_marginal_out.append(marginal_out)


    return {
        "G_reuse": np.stack(G_reuse),
        "G_new": np.stack(G_new),
        "G_marginal": np.stack(G_marginal),
        "G_marginal_in": np.stack(G_marginal_in),
        "G_marginal_out": np.stack(G_marginal_out)
    }


def compute_stagewise_IO_changes(analyzer: DynamicsAnalyzer, 
                                 max_rank=MAX_RANK
    ):

    try: 
        learning_events = detect_learning_windows(analyzer)

    except Exception as e:
        print(
            e.message,
            e.args
        )

    try:
        dW_in = deepcopy(analyzer.dW_in)
        dW_out = deepcopy(analyzer.dW_out)

    except Exception as e:
        print(
            e.message,
            e.args
        )

    centres = np.asarray(
        [
        event["center_idx"] 
        for event in learning_events["events"]
        ]
    )

    U_rec_at_centers = analyzer.rec_U[centres]
    V_rec_at_centers = analyzer.rec_V[centres] 
    
    if dW_in.shape[1] != U_rec_at_centers.shape[1]:
        dW_in = np.swapaxes(dW_in, 1, 2)

    if dW_out.shape[1] != U_rec_at_centers.shape[1]:
        dW_out = np.swapaxes(dW_out, 1, 2)

    num_inputs = dW_in.shape[-1]
    num_outputs = dW_out.shape[-1]


    # make nested lists for each input and output direction (not so readable, but extends nicely to different set-ups)
    dWin_cumulative = [[] for _ in range(num_inputs)]
    dWin_marginal = [[] for _ in range(num_inputs)]

    dWout_cumulative = [[] for _ in range(num_outputs)]
    dWout_marginal = [[] for _ in range(num_outputs)]




    for idx, (start, end) in enumerate(zip(centres[:-1], centres[1:])):

        
        U = U_rec_at_centers[idx]
        V = V_rec_at_centers[idx]

        Win = dW_in[start:end]
        Wout = dW_out[start:end]


        for i in range(end - start):

            for j in range(num_inputs):

                cumulative, marginal = vector_energy_spectrum(
                    Win[i, :, j],
                    U,
                    max_rank
                )

                dWin_cumulative[j].append(
                    cumulative
                )

                dWin_marginal[j].append(
                    marginal
                )

            for j in range(num_outputs):

                cumulative, marginal = vector_energy_spectrum(
                    Wout[i, :, j],
                    V,
                    max_rank
                )

                dWout_cumulative[j].append(
                    cumulative
                )

                dWout_marginal[j].append(
                    marginal
                )
                
    dWin_cumulative = np.stack(
        [
            np.stack(x)
            for x in dWin_cumulative
        ],
        axis=1
    )

    dWin_marginal = np.stack(
        [
            np.stack(x)
            for x in dWin_marginal
        ],
        axis=1
    )


    dWout_cumulative = np.stack(
        [
            np.stack(x)
            for x in dWout_cumulative
        ],
        axis=1
    )

    dWout_marginal = np.stack(
        [
            np.stack(x)
            for x in dWout_marginal
        ],
        axis=1
    )

    return {
        "dWin_cumulative": dWin_cumulative,
        "dWin_marginal": dWin_marginal,
        "dWout_cumulative": dWout_cumulative,
        "dWout_marginal": dWout_marginal
    }


    






