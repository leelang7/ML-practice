import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 데이터를 생성하고 반환하는 함수입니다.
def load_data():
    np.random.seed(0)
    X = np.random.normal(size = 100)
    print(X.shape)
    y = (X > 0).astype(np.float)
    X[X > 0] *= 5
    X += .7 * np.random.normal(size = 100)
    X = X[:, np.newaxis]
    print(X.shape)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 100)

    return train_X, test_X, train_y, test_y

"""
1. 로지스틱 회귀 모델을 구현하고, 
   학습 결과를 확인할 수 있는 main() 함수를 완성합니다. 
"""

def main():
 

    # 예측 결과 확인하기 
    print("예측 결과 :", predicted[:10])
    


    return logistic_model

if __name__ == "__main__":
    main()

