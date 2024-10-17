import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터를 생성하고 반환하는 함수입니다.
def load_data():
    np.random.seed(0)
    X = np.random.normal(size = 100)
    y = (X > 0).astype(float) # np.float  => float
    X[X > 0] *= 5
    X += .7 * np.random.normal(size = 100)
    X = X[:, np.newaxis]
    print(X.shape, X[0])
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 100)
    return train_X, test_X, train_y, test_y

"""
1. 로지스틱 회귀 모델을 구현하고, 
   학습 결과를 확인할 수 있는 main() 함수를 완성합니다. 
"""
def plot_logistic_regression(model, train_X, train_y):
    # Create a range of values for prediction
    x_range = np.linspace(train_X.min(), train_X.max(), 300)[:, np.newaxis]
    y_prob = model.predict_proba(x_range)[:, 1]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(train_X, train_y, color='blue', label='Training data')
    plt.plot(x_range, y_prob, color='red', linewidth=2, label='Logistic Regression Model')
    plt.title('Logistic Regression Visualization')
    plt.xlabel('Feature X')
    plt.ylabel('Probability')
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
    plt.legend()
    plt.grid()
    plt.show()

def main():
    train_X, test_X, train_y, test_y = load_data()
    logistic_model = LogisticRegression() 
    logistic_model.fit(train_X, train_y)
    predicted = logistic_model.predict(test_X)
    
    # 예측 결과 확인하기 
    print("예측 결과 :", predicted[:10])
    
    plot_logistic_regression(logistic_model, train_X, train_y)

    return logistic_model

if __name__ == "__main__":
    main()
