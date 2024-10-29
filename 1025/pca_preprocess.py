import numpy as np
import csv

def load_data():
    with open('./data/iris.csv', 'r', encoding='utf-8') as f:
        raw_data = csv.reader(f)
        sepal_length = []
        sepal_width = []
        petal_length = []
        petal_width = []

        next(raw_data)  # 첫 번째 행을 건너뜀

        for line in raw_data:
            sepal_length.append(line[0])
            sepal_width.append(line[1])
            petal_length.append(line[2])
            petal_width.append(line[3])

        sepal_length = np.array(sepal_length).astype(np.float64)
        sepal_width = np.array(sepal_width).astype(np.float64)
        petal_length = np.array(petal_length).astype(np.float64)
        petal_width = np.array(petal_width).astype(np.float64)

        return np.stack((sepal_length, sepal_width, petal_length, petal_width), axis=1)
