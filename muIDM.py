
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
    A_eigvals, A_eigvecs = scipy.sparse.linalg.eigsh(A, k=n_components)
    A_eigvals = np.flip(A_eigvals)
    A_eigvecs = np.flip(A_eigvecs, axis=1)

    dm_embedding = Q_inv_sqrt @ A_eigvecs @ np.diag(A_eigvals)

    return dm_embedding

def nystrom_DM(X, S, S_complement, n_neighbors=5, rw_laziness=0.5, t=5):
    K = graphtools.api.Graph(X, knn=n_neighbors).to_pygsp().W.tocsr()
    K = 1/2 * (K + K.T)

    degrees = np.array(K.sum(axis=0))[0]
    
    Q = scipy.sparse.diags(degrees)
    Q_inv = scipy.sparse.diags(1/degrees)
    Q_sqrt = scipy.sparse.diags(degrees**(1/2))
    Q_inv_sqrt = scipy.sparse.diags(degrees**(-1/2))
    print(Q_inv_sqrt)

    P = Q_inv @ K
    P = (1 - rw_laziness) * P + rw_laziness * scipy.sparse.eye(P.shape[0])
    P = np.linalg.matrix_power(P.todense(), t)
    A = Q_sqrt @ P @ Q_inv_sqrt
    A_sub_ul = A[np.ix_(S, S)] # A upper-left submatrix
    A_sub_ur = A[np.ix_(S, S_complement)]

    A_sub_ul_eigvals, A_sub_ul_eigvecs = scipy.linalg.eigh(A_sub_ul, driver='ev')
    A_sub_ul_eigvals = np.flip(A_sub_ul_eigvals)
    A_sub_ul_eigvecs = np.flip(A_sub_ul_eigvecs, axis=1)
    nystrom = A_sub_ur.T @ A_sub_ul_eigvecs @ np.diag(1 / A_sub_ul_eigvals)
    approx_eigvecs = np.concatenate((A_sub_ul_eigvecs, nystrom), axis=0)
    dm_embedding = Q_inv_sqrt @ approx_eigvecs @ np.diag(A_sub_ul_eigvals)

    return dm_embedding

def ONM(A, degrees, S, S_complement, rw_laziness=0.75):
    Q_inv_sqrt = scipy.sparse.diags(degrees**(-1/2))

    A_tilde = A[S]
    A_sub_ul = A[np.ix_(S, S)] # A upper-left submatrix
    A_sub_ur = A[np.ix_(S, S_complement)]

    A_sub_ul_eigvals, A_sub_ul_eigvecs = scipy.linalg.eigh(A_sub_ul, driver='ev')
    A_sub_ul_inv_sqrt = A_sub_ul_eigvecs @ np.diag(A_sub_ul_eigvals**(-1/2)) @ A_sub_ul_eigvecs.T

    C = A_sub_ul + A_sub_ul_inv_sqrt @ A_sub_ur @ A_sub_ur.T @ A_sub_ul_inv_sqrt
    C_u, C_s, _ = np.linalg.svd(C, hermitian=True)
    Delta = scipy.sparse.diags(C_s)
    onm_embedding = Q_inv_sqrt @ A_tilde.T @ A_sub_ul_inv_sqrt @ C_u @ np.sqrt(Delta)

    return onm_embedding

def compute_T(old_onm_embedding, new_onm_embedding, S):

    return T 

def muIDM(X, mu, n_neighbors=40, rw_laziness=0.5, t=5):
    """
    Parameters
    ----------
    X: {array-like} of shape (n_samples, n_features)
        Data matrix.

    mu: float
        Distance error bound (using the l-inf norm) for approximation.

    n_neighbors: int
        Number of nearest neighbors for nearest neighbors graph building.

    rw_laziness: float
        Probability that a random walker remains at its current vertex in each time step.

    t: int
        Power that the diffusion operator is raised to.

    Returns
    ----------
    Matrix of shape (n_samples, n_components) containing the approximate diffusion coordinates,
    where n_components is determined by the algorithm given a value of mu.
    """
    
    K = graphtools.api.Graph(X, decay=None, knn=n_neighbors).to_pygsp().W.tocsr()
    K = 1/2 * (K + K.T)

    degrees = np.array(K.sum(axis=0))[0] 
    Q = scipy.sparse.diags(degrees)
    Q_inv = scipy.sparse.diags(1/degrees)
    Q_sqrt = scipy.sparse.diags(degrees**(1/2))
    Q_inv_sqrt = scipy.sparse.diags(degrees**(-1/2))

    P = Q_inv @ K
    P = (1 - rw_laziness) * P + rw_laziness * scipy.sparse.eye(P.shape[0])
    P = np.linalg.matrix_power(P.todense(), t)
    A = Q_sqrt @ P @ Q_inv_sqrt

    S = [0]
    S_complement = list(range(1, X.shape[0]))
    onm_embedding = ONM(A, degrees, S, S_complement, rw_laziness)
    change_of_basis = np.linalg.inv(onm_embedding[S]) 

    i = 0
    while i < len(S_complement):
        k = S_complement[i]
        new_onm_embedding = ONM(A,
                                degrees,
                                S + [k],
                                S_complement[:i] + S_complement[i + 1:], 
                                rw_laziness)

        T = change_of_basis @ new_onm_embedding[S]
        beta = np.max(np.abs(new_onm_embedding[k] - (onm_embedding[k] @ T)))
        if beta > mu / 2:
            onm_embedding = new_onm_embedding
            S.append(k)
            S_complement = S_complement[:i] + S_complement[i + 1:]
            change_of_basis = np.linalg.inv(onm_embedding[S]) 
        else:
            i += 1
    
    return onm_embedding
   
def test_muIDM_on_swiss_roll(n_samples, mu):
    X, t = sklearn.datasets.make_swiss_roll(n_samples)
    onm = muIDM(X, n_neighbors=int(n_samples/2), rw_laziness=0.75, mu=mu)
    scprep.plot.scatter3d(X, c=onm[:,1])
    plt.show() 

if __name__=='__main__':
    test_muIDM_on_swiss_roll(500, 0.01)

