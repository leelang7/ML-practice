import numpy as np
from scipy.linalg import svd

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.lambda_ = None  # 고유값을 저장
        self.V = None  # 고유 벡터를 저장
        self.mean_ = None  # 데이터 포인트의 평균을 저장
        self.W = None  # 변환 행렬을 저장
        self.original_data = None  # 원본 데이터를 저장

    def fit_transform(self, X):
        self.original_data = X  # 원본 데이터를 클래스의 속성으로 저장
        self.mean_ = X.mean(axis=0)
        diff = X - self.mean_
        cov = diff.T.dot(diff) / (X.shape[0] - 1)
        self.V, self.lambda_, _ = svd(cov)
        order = (-self.lambda_).argsort()
        self.W = self.V[:, order[:self.n_components]]
        return (X - self.mean_).dot(self.W)

    def inverse_transform(self, X_hat):
        return X_hat.dot(self.W.T) + self.mean_
