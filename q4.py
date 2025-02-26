import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

"""
1. 사이킷런에 저장된 데이터를 불러오고, 
   2개의 변수만을 가질 수 있도록 
   고정하여 반환하는 함수를 구현합니다.
"""
def load_data():
    
    X, y = None
    
    column_start = 6
    X = X[:, column_start : column_start+2]
    print(X.shape)
    return X
    
"""
2. 주성분 분석(PCA)을 수행하여 
   2차원 데이터를 1차원으로 축소하는 함수를 완성합니다.
"""
def pca_data(X):
    
    pca = None
    
    pca.None
    
    X_pca = None
    
    return pca, X_pca

# 축소된 주성분 축과 데이터 산점도를 그려주는 함수입니다.
def visualize(pca, X, X_pca):
    X_new = pca.inverse_transform(X_pca)
    
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    plt.axis('equal');
    
    plt.savefig('PCA.png')

def main():
    
    X = load_data()
    
    pca, X_pca = pca_data(X)
    
    print("- original shape:   ", X.shape)
    print("- transformed shape:", X_pca.shape)
    
    visualize(pca, X, X_pca)
    
if __name__ == '__main__':
    main()
