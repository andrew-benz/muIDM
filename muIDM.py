
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

def ONM(X, n_components=2, n_neighbors=5):
    pass

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
    #A_eigvals, A_eigvecs = scipy.sparse.linalg.eigsh(A, k=n_components)
    A_eigvals, A_eigvecs = np.linalg.eigh(A.todense())
    A_eigvals = np.flip(A_eigvals)
    #print(A_eigvals)
    A_eigvecs = np.flip(A_eigvecs, axis=1)
    dm_embedding = Q_inv_sqrt @ A_eigvecs @ np.diag(A_eigvals)

    # TODO: remove this
    test_cutoff = 500
    S = list(range(test_cutoff))
    S_complement = list(range(test_cutoff + 1, X.shape[0]))

    A_tilde = A[S]
    #print(A_tilde.shape)
    A_sub_ul = A[np.ix_(S, S)] # A upper-left submatrix
    A_sub_ur = A[np.ix_(S, S_complement)]

    #print(A_sub_ul.toarray())
    A_sub_ul_eigvals, A_sub_ul_eigvecs = scipy.linalg.eigh(A_sub_ul.toarray(), driver='ev')
    #print(A_sub_ul_eigvals)
    A_sub_ul_eigvals = np.flip(A_sub_ul_eigvals)
    A_sub_ul_eigvecs = np.flip(A_sub_ul_eigvecs, axis=1)
    A_sub_ul_inv_sqrt = A_sub_ul_eigvecs @ np.diag(A_sub_ul_eigvals**(-1/2)) @ A_sub_ul_eigvecs.T

    C = A_sub_ul + A_sub_ul_inv_sqrt @ A_sub_ur @ A_sub_ur.T @ A_sub_ul_inv_sqrt
    #print(C)
    #print(C - C.T)
    C_eigvals, C_eigvecs = scipy.linalg.eigh(C)
    C_eigvals = np.flip(C_eigvals)
    C_eigvecs = np.flip(C_eigvecs, axis=1)
    Delta = scipy.sparse.diags(C_eigvals)
    onm_embedding = Q_inv_sqrt @ A_tilde.T @ A_sub_ul_inv_sqrt @ C_eigvecs @ np.sqrt(Delta)

    return onm_embedding, dm_embedding


X, t = sklearn.datasets.make_swiss_roll(2000)
onm, dm = muIDM(X)
print(np.around(dm, decimals=4))
print(np.around(onm, decimals=4))
i = np.argmax(np.linalg.norm(onm, axis=0))
#print(onm.shape)
#print(np.around(sklearn.metrics.pairwise.pairwise_kernels(dm), 2))
#print(np.around(sklearn.metrics.pairwise.pairwise_kernels(onm), 2))
#print(sklearn.metrics.pairwise.pairwise_kernels(dm)[:3, :3])
#print(sklearn.metrics.pairwise.pairwise_kernels(onm))
#print(sklearn.metrics.pairwise.pairwise_kernels(dm_2))
#print(onm.shape)
#print(onm)
scprep.plot.scatter3d(X, c=onm[:,1])
#scprep.plot.scatter3d(muIDM(X).T[:,:3])
plt.show()
    
