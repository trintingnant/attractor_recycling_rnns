from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.manifold import SpectralEmbedding
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import get_colors

def pairwise_cca(
        X, 
        Y,
        n_components=10,
        cca_dim=None,
        max_iter=10000,
        tol=1e-6
    ):

    # assume for the minute same number of neurons
    n_cond, T, num_neurons = X.shape

    X_flattened = X.reshape(n_cond * T, num_neurons)
    Y_flattened = Y.reshape(n_cond * T, num_neurons)

    dim = min(n_components,
            X_flattened.shape[0],
            X_flattened.shape[1],
            Y_flattened.shape[0],
            Y_flattened.shape[1]
            )
    
    X_PCA = PCA(n_components=dim)
    Y_PCA = PCA(n_components=dim)

    X_fit = X_PCA.fit_transform(X_flattened)
    Y_fit = Y_PCA.fit_transform(Y_flattened)

    if cca_dim is None:
        cca_dim = dim

    d = min(cca_dim, dim)
    cca = CCA(
        n_components=d,
        scale=False,
        max_iter=max_iter,
        tol=tol
        )
    
    cca.fit(X_fit, Y_fit)

    X_CCA, Y_CCA = cca.transform(X_fit, Y_fit)

    # get canonical correlations 

    ccorrs = np.array([
                np.corrcoef(X_CCA[:,idx], Y_CCA[:, idx]) [0, 1]
                for idx in range(d)
    ])
    
    # now get the CCA maps

    X_map = X_PCA.components_.T @ cca.x_rotations_
    Y_map = Y_PCA.components_.T @ cca.y_rotations_

    X_CCA_full = X_CCA.reshape(n_cond, T, d)
    Y_CCA_full = Y_CCA.reshape(n_cond, T, d)

    return ({"pca_x_": X_PCA,
            "pca_y_": Y_PCA,
            "cca_": cca,

            "x_fit_": X_fit,
            "y_fit_": Y_fit,

            "x_cca_": X_CCA,
            "y_cca_": Y_CCA,

            "x_cca_traj_": X_CCA_full, 
            "y_cca_traj_":Y_CCA_full,

            "corr_coeff_": ccorrs, 

            "x_map_": X_map,
            "y_map_": Y_map,
    })

def get_spectral_embedding(S):

    """
    Code for obtaining projections of the graph defined
    by the CCA similarities via a spectral embedding
    (Like PCA but we eigen-decompose the graph Laplacian)
    """

    # Prepare affinity matrix
    A = np.asarray(S).copy()

    # Symmetrize
    A = 0.5 * (A + A.T)

    # Affinities must be non-negative
    A = np.clip(A, 0, None)

    # Remove trivial self-similarity
    np.fill_diagonal(A, 0)

    # Spectral embedding
    embedding = SpectralEmbedding(
        n_components=2,
        affinity="precomputed",
        random_state=0
    )

    X = embedding.fit_transform(A)
    
    return X


# For the spectral decomposition we also want to add out-of-sample extensions
# to be able to see where our reverse-engineered models live in comparison to 
# the trained-full rank models (we are using the low-rank distils of these
# so there shouldn't be a big problem)
# however, the space should the same space (!). So we need the OOS-Extension

def prepare_affinity_matrix(S):

    A = np.asarray(S).copy()
    A = 0.5 * (A + A.T)
    A = np.clip(A, 0, None)
    np.fill_diagonal(A, 0)

    return A

def cca_affinities_to_reference(
    X_new,
    reference_rates,
    n_components=10,
    cca_dim=None,
    max_iter=10000,
    tol=1e-6
):

    s_new = np.zeros(len(reference_rates), dtype=float)

    for idx, X_ref in enumerate(reference_rates):

        cca_result = pairwise_cca(
            X_new,
            X_ref,
            n_components=n_components,
            cca_dim=cca_dim,
            max_iter=max_iter,
            tol=tol
        )

        # scalar similarity used in S
        s_new[idx] = np.mean(
            cca_result["corr_coeff_"]
        )

    return s_new


def cca_affinities_many_to_reference(
    new_rates,
    reference_rates,
    n_components=10,
    cca_dim=None,
    max_iter=10000,
    tol=1e-6
    ):
    
    return np.stack([
        cca_affinities_to_reference(
            X_new,
            reference_rates,
            n_components=n_components,
            cca_dim=cca_dim,
            max_iter=max_iter,
            tol=tol
        )
        for X_new in new_rates
    ])


def make_extend_state(A, X):

    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square affinity matrix.")

    if X.ndim != 2 or X.shape[0] != A.shape[0]:
        raise ValueError(
            "X must have one row for every row/column of A."
        )

    degree = A.sum(axis=1)

    if np.any(degree <= 0):
        raise ValueError(
            "Affinity graph contains at least one zero-degree node."
        )

    M = A / np.sqrt(
        degree[:, None] * degree[None, :]
    )

    # Recover normalized-affinity eigenvectors corresponding to X.
    U = np.sqrt(degree[:, None]) * X

    mu = np.zeros(X.shape[1], dtype=float)

    for k in range(X.shape[1]):

        u = U[:, k]

        denominator = u.T @ u

        if np.isclose(denominator, 0):
            raise ValueError(
                f"Spectral component {k} has zero norm."
            )

        mu[k] = (u.T @ M @ u) / denominator

    if np.any(np.isclose(mu, 0)):
        raise ValueError(
            "At least one retained spectral component has an "
            "eigenvalue too close to zero for stable Nyström extension."
        )

    return {
        "A": A,
        "X": X,
        "degree": degree,
        "mu": mu
    }


def make_nystrom_state_from_similarity(S, X):
    """
    Convenience wrapper: preprocess S exactly as in the original
    plotting function, then construct the Nyström state.
    """

    A = prepare_affinity_matrix(S)
    return make_extend_state(A, X)



def extend(s_new, spectral_state):
    """
    Coords for the OOS odels:

        x_new,k =
            [sum_j A_new,j X_j,k]
            / [mu_k d_new],

        d_new = sum_j A_new,j.

    """

    s_new = np.asarray(s_new, dtype=float)

    single_point = (s_new.ndim == 1)

    if single_point:
        s_new = s_new[None, :]

    if s_new.ndim != 2:
        raise ValueError(
            "s_new must have shape (N,) or (n_new, N)."
        )

    s_new = np.clip(s_new, 0, None)

    X = spectral_state["X"]
    mu = spectral_state["mu"]


    d_new = s_new.sum(axis=1)

    X_new = (
        s_new @ X
        /
        (d_new[:, None] * mu[None, :])
    )

    if single_point:
        return X_new[0]

    return X_new


def plot_heldout_points(
    ax,
    X_new,
    colors=None,
    labels=None,
    marker="*",
    s=200,
    edgecolor="black",
    linewidth=1.5,
    zorder=10
    ):

    X_new = np.asarray(X_new)

    if X_new.ndim == 1:
        X_new = X_new[None, :]

    if X_new.shape[1] != 2:
        raise ValueError(
            "plot_heldout_points expects a 2D spectral embedding."
        )

    n_new = X_new.shape[0]

    if labels is None:
        labels = [None] * n_new
    elif isinstance(labels, str):
        labels = [labels]
    else:
        labels = list(labels)

    if len(labels) != n_new:
        raise ValueError(
            "labels must contain one label per held-out model."
        )

    if colors is None:
        colors = [None] * n_new
    else:
        colors_array = np.asarray(colors)

        # A single RGB/RGBA triple should be treated as one color,
        # not as one color per model.
        if (
            colors_array.ndim == 1
            and colors_array.size in (3, 4)
            and np.issubdtype(colors_array.dtype, np.number)
        ):
            colors = [tuple(colors_array)] * n_new

        elif isinstance(colors, str):
            colors = [colors] * n_new

        else:
            colors = list(colors)

            if len(colors) == 1 and n_new > 1:
                colors = colors * n_new

    if len(colors) != n_new:
        raise ValueError(
            "colors must contain one color per held-out model, or "
            "one RGB/RGBA color to use for all held-out models."
        )

    for idx in range(n_new):

        scatter_kwargs = dict(
            x=[X_new[idx, 0]],
            y=[X_new[idx, 1]],
            s=s,
            marker=marker,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
            label=labels[idx]
        )

        if colors[idx] is not None:
            scatter_kwargs["c"] = [colors[idx]]

        ax.scatter(**scatter_kwargs)

    return ax



