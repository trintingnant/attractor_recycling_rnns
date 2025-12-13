"""
Clusters LR RNN using Guassian Mixtures
Adapted from Dubreuil, Valente et al. 2022
"""
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import adjusted_rand_score
import os
import pickle as pkl

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

def clustering_stability(X, model, n_iter, random_state=None):

    """
    Computes the stability of clusterings.
    Obtained by bootstrapping the connectivity
    parameters and re-applying the GMM to the 
    bootstrapped subsamples 'n_iter' times.

    adapted from: https://amueller.github.io/aml/04-model-evaluation/17-cluster-evaluation.html

    """

    from sklearn.base import clone

    labels = []
    idxs = []
    rng = np.random.RandomState(1)
    for i in range(n_iter):
        sub_idxs = rng.randint(0, X.shape[0], X.shape[0])
        idxs.append(sub_idxs)
        model_new = clone(model)
        model_new.random_state = rng.randint(1e6)

        X_sub = X[sub_idxs]
        model_new.fit(X_sub)

        labels_new = -np.ones(X.shape[0], dtype=int)
        labels_new[sub_idxs] = model_new.predict(X_sub)
        labels.append(labels_new)

    scores = []
    for l, i in zip(labels, idxs):
        for k, j in zip(labels, idxs):
            inter_ = np.intersect1d(i,j)
            score = adjusted_rand_score(l[inter_], k[inter_])
            scores.append(score)

    return np.mean(scores)

    




    




