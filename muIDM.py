
import numpy as np
import scipy
import scipy.sparse

import sklearn
import sklearn.base
import sklearn.neighbors
import sklearn.datasets
import sys

import graphtools

import scprep
import matplotlib.pyplot as plt


def DM(X, n_components=4, n_neighbors=5, rw_laziness=0.5):
    K = graphtools.api.Graph(X).to_pygsp().W.tocsr()

    degrees = np.array(K.sum(axis=0))[0]
    
    Q = scipy.sparse.diags(degrees)
    Q_inv = scipy.sparse.diags(1/degrees)
    Q_sqrt = scipy.sparse.diags(degrees**(1/2))
    Q_inv_sqrt = scipy.sparse.diags(degrees**(-1/2))

    P = Q_inv @ K
    P = (1 - rw_laziness) * P + rw_laziness * scipy.sparse.eye(P.shape[0])
    A = Q_sqrt @ P @ Q_inv_sqrt
    A_eigvals, A_eigvecs = scipy.sparse.linalg.eigsh(A, k=n_components)
    A_eigvals = np.flip(A_eigvals)
    A_eigvecs = np.flip(A_eigvecs, axis=1)

    dm_embedding = Q_inv_sqrt @ A_eigvecs @ np.diag(A_eigvals)

    return dm_embedding

def ONM(X, aff_matrix, S, S_complement, rw_laziness=0.75):
    #K = sklearn.neighbors.kneighbors_graph(X, n_neighbors)
    K = graphtools.api.Graph(X, knn=n_neighbors).to_pygsp().W.tocsr()
    K = 1/2 * (K + K.T)

    degrees = np.array(K.sum(axis=0))[0] 
    Q = scipy.sparse.diags(degrees)
    Q_inv = scipy.sparse.diags(1/degrees)
    Q_sqrt = scipy.sparse.diags(degrees**(1/2))
    Q_inv_sqrt = scipy.sparse.diags(degrees**(-1/2))

    P = Q_inv @ K
    P = (1 - rw_laziness) * P + rw_laziness * scipy.sparse.eye(P.shape[0])
    A = Q_sqrt @ P @ Q_inv_sqrt

    A_tilde = A[S]
    A_sub_ul = A[np.ix_(S, S)] # A upper-left submatrix
    A_sub_ur = A[np.ix_(S, S_complement)]

    A_sub_ul_eigvals, A_sub_ul_eigvecs = scipy.linalg.eigh(A_sub_ul.toarray(), driver='ev')
    #A_sub_ul_eigvals = np.flip(A_sub_ul_eigvals)
    #A_sub_ul_eigvecs = np.flip(A_sub_ul_eigvecs, axis=1)
    A_sub_ul_inv_sqrt = A_sub_ul_eigvecs @ np.diag(A_sub_ul_eigvals**(-1/2)) @ A_sub_ul_eigvecs.T

    C = A_sub_ul + A_sub_ul_inv_sqrt @ A_sub_ur @ A_sub_ur.T @ A_sub_ul_inv_sqrt
    C_u, C_s, _ = np.linalg.svd(C, hermitian=True)
    Delta = scipy.sparse.diags(C_s)
    onm_embedding = Q_inv_sqrt @ A_tilde.T @ A_sub_ul_inv_sqrt @ C_u @ np.sqrt(Delta)

    return onm_embedding


def muIDM(X, n_components=2, n_neighbors=40, rw_laziness=0.5, mu=1):
    """
    Parameters
    ----------
    X: {array-like} of shape (n_samples, n_features)
        Data matrix.

    n_components: int
        The dimension of the projected subspace.

    n_neighbors: int
        Number of nearest neighbors for nearest neighbors graph building.

    mu: float
        Distance error bound for approximation.

    gamma: float
        Kernel coefficient for rbf kernel.

    Returns
    ----------
    Matrix of shape (n_samples, n_components) containing the approximate diffusion coordinates.
    """

    pass

n_samples = 3000
n_neighbors = 70
X, t = sklearn.datasets.make_swiss_roll(n_samples)
K = graphtools.api.Graph(X, knn=n_neighbors).to_pygsp().W.tocsr()

dict_size = int(4/5 * n_samples)
K = 1/2 * (K + K.T)
onm = ONM(X, K, list(range(dict_size)), list(range(dict_size, n_samples)))

scprep.plot.scatter3d(onm[:,1:4], t)
plt.show()
    
