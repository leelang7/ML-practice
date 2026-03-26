import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 유방암 데이터를 불러오고,학습용 데이터와 테스트용 데이터로 분리하여 반환하는 함수입니다.
def load_data():
    
    X, y = load_breast_cancer(return_X_y = True)
    
    train_X, test_X, train_y ,test_y = train_test_split(X, y, test_size = 0.2, random_state = 156)
    
    return train_X, test_X, train_y ,test_y
"""
1. 다양한 모델을 사용하는 VotingClassifier를 정의하여
   학습시키고, 예측을 수행한 결과를 반환하는 함수를 구현합니다.
"""
def Voting_Clf(train_X, test_X, train_y ,test_y):
    
    lr_clf = None
    knn_clf = None
    
    vo_clf = None
    
    None
    
    pred = None
    
    return lr_clf, knn_clf, vo_clf, pred
    
# 데이터를 불러오고, 모델 학습 및 예측을 진행하기 위한 함수입니다.
def main():
    
    train_X, test_X, train_y ,test_y = load_data()
    
    lr_clf, knn_clf,vo_clf, pred = Voting_Clf(train_X, test_X, train_y ,test_y)
    
    print('> Voting Classifier 정확도 : {0:.4f}\n'.format(accuracy_score(test_y, pred)))
    
    # 다른 분류기를 각각 학습했을 때 결과 예측
    classifiers = [lr_clf, knn_clf]
    for classifier in classifiers:
        classifier.fit(train_X, train_y)
        pred = classifier.predict(test_X)
        class_name = classifier.__class__.__name__
        print("> {0} 정확도 : {1:.4f}".format(class_name, accuracy_score(test_y, pred)))

if __name__ =="__main__":
    main()