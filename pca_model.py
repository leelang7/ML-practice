import numpy as np
from scipy.linalg import svd

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.lambda_ = None  # Eigenvalues will be stored here
        self.V = None  # Eigenvectors will be stored here
        self.mean_ = None  # Mean of the data points will be stored here
        self.W = None  # Transformation matrix will be stored here

    def fit_transform(self, X):
        self.mean_ = X.mean(axis=0)
        diff = X - self.mean_
        cov = diff.T.dot(diff) / (X.shape[0] - 1)
        U, S, Vt = svd(cov)
        self.lambda_ = S
        self.V = Vt.T
        order = (-self.lambda_).argsort()
        self.W = self.V[:, order[:self.n_components]]
        return (X - self.mean_).dot(self.W)

    def inverse_transform(self, X_hat):
        return X_hat.dot(self.W.T) + self.mean_
