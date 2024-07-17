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

def plot_logistic_regression(model, X, y):
    # X는 feature가 하나이므로 간단히 시각화가 가능합니다.
    plt.figure(figsize=(10, 6))
    
    # 데이터 포인트 그리기
    plt.scatter(X, y, color='black', zorder=20)
    
    # 결정 경계 그리기
    X_test = np.linspace(-5, 25, 300)
    loss = model.predict_proba(X_test[:, np.newaxis])[:, 1]
    plt.plot(X_test, loss, color='red', linewidth=3)
    
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(1, color='black', linestyle='--')
    plt.axhline(0.5, color='blue', linestyle='--')
    plt.axvline(0, color='black', linestyle='--')

    plt.ylabel('Probability')
    plt.xlabel('Feature Value')
    plt.title('Logistic Regression - Probability vs Feature')
    plt.show()

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

