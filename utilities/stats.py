# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import linalg as LA

# stats methods for matrix similarity:
# http://math.stackexchange.com/questions/690972/
#   distance-or-similarity-between-matrices-that-are-not-the-same-size
def rv_coefficient(X, Y):
    min_denominator = 0.000000000001
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    nom = np.trace(X.dot(X.T).dot(Y).dot(Y.T))
    den = np.sqrt(np.trace(X.dot(X.T))**2 * np.trace(Y.dot(Y.T))**2)
    return nom / den if den > min_denominator else 1


def vec(X):
    return np.reshape(X.T, (-1, 1))


def rv2_coefficient(X, Y):
    min_denominator = 0.000000000001
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    # https://academic.oup.com/bioinformatics/article/25/3/401/244239/Matrix-correlations-for-high-dimensional-data-the
    vecX = vec(X.dot(X.T) - np.diag(X.dot(X.T)))
    vecY = vec(Y.dot(Y.T) - np.diag(Y.dot(Y.T)))
    nom = vecX.T.dot(vecY)
    den = np.sqrt(vecX.T.dot(vecX) * vecY.T.dot(vecY))
    return np.average(nom / den) if den > min_denominator else 1


def distance_correlation(X, Y):
    min_denominator = 0.000000000001
    # https://gist.github.com/satra/aa3d19a12b74e9ab7941
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    nom = np.sqrt(dcov2_xy)
    den = np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return nom / den if den > min_denominator else 1

# source http://stackoverflow.com/a/13224592/7410738
# the sklearn PCA does not allow for more components than samples
def pca(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T, evals, evecs