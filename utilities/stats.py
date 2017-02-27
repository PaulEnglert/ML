# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import pdist, squareform


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
