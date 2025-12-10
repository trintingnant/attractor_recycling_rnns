"""
Clusters LR RNN using Guassian Mixtures
Adapted from Dubreuil, Valente et al. 2022
"""

import numpy as np
from scipy import stats
import numpy as np
from scipy import stats
import torch
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import SpectralClustering
from rnn import RNN, load_rnn
from utils import phiPrime
import os
import pickle as pkl
from Tasks.TaskManager import Task
from Tasks.cycling import Cycling


def fit_gmm(
        X,
        n_components,
        n_init=25,
        random_state=None,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=None,
        mean_precision_prior=None
):

    model = BayesianGaussianMixture(
            n_components=n_components,
            n_init=n_init,
            random_state=random_state,
            init_params="random",
            mean_precision_prior=mean_precision_prior,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=weight_concentration_prior,
        )
    model.fit(X)
    label = model.predict(X)

    return label, model 
