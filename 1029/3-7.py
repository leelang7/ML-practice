import pca_preprocess as pre
import pca_model as md
import numpy as np

def variance_calculator(model):
    # 원본 데이터의 분산
    original_var = np.sum(np.var(model.original_data, axis=0, ddof=1))
    print("원래 데이터 분산(전체):", original_var)

    # 축소된 데이터의 분산
    shrinked_var = np.sum(model.lambda_[:model.n_components])
    print("감소된 데이터 분산(주요 구성 요소):", shrinked_var)

    # 분산 비율
    variance_percentage = shrinked_var / original_var
    return variance_percentage

def your_choice(X):
    myPCA = md.PCA(n_components=2)
    myPCA.fit_transform(X)  # PCA 학습 및 변환
    return myPCA

def main():
    X = pre.load_data()
    myPCA = your_choice(X)
    Percent = variance_calculator(myPCA)
    print('원본 데이터의 차원 = {}, n_components = {}, 원본 대비 축소 후의 분산 비율: {:.2f}'.format(X.shape[1], myPCA.n_components, Percent))

if __name__ == "__main__":
    main()
