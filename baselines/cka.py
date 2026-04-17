"""
CKA (Centered Kernel Alignment) baseline.
Linear CKA and RBF-kernel CKA, following Kornblith et al. 2019.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def center_gram(K):
    """Center a Gram matrix: H K H where H = I - 1/n 11^T."""
    n = K.shape[0]
    ones = np.ones((n, n)) / n
    return K - ones @ K - K @ ones + ones @ K @ ones


def hsic(K, L):
    """Unbiased HSIC estimator (Song et al. 2012)."""
    Kc = center_gram(K)
    Lc = center_gram(L)
    n = K.shape[0]
    return np.trace(Kc @ Lc) / ((n - 1) ** 2)


def linear_cka(X, Y):
    """Linear CKA: HSIC(XX^T, YY^T) / sqrt(HSIC(XX^T, XX^T) * HSIC(YY^T, YY^T))."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    Kx = X @ X.T
    Ky = Y @ Y.T
    hsic_xy = hsic(Kx, Ky)
    hsic_xx = hsic(Kx, Kx)
    hsic_yy = hsic(Ky, Ky)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def rbf_cka(X, Y):
    """RBF-kernel CKA with median bandwidth heuristic."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    def rbf_gram(Z):
        sq = pdist(Z, metric='sqeuclidean')
        med = np.median(sq)
        if med < 1e-10:
            med = 1.0
        gamma = 1.0 / med
        return np.exp(-gamma * squareform(sq))

    Kx = rbf_gram(X)
    Ky = rbf_gram(Y)
    hsic_xy = hsic(Kx, Ky)
    hsic_xx = hsic(Kx, Kx)
    hsic_yy = hsic(Ky, Ky)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)
