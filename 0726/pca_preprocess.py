import numpy as np
import csv

def load_data():
    with open('./data/iris.csv', 'r', encoding='utf-8') as f:
        raw_data = csv.reader(f)
        sepal_length = []
        sepal_width = []
        petal_length = []
        petal_width = []
        
        for lines in raw_data:
            sepal_length.append(lines[0])
            sepal_width.append(lines[1])
            petal_length.append(lines[2])
            petal_width.append(lines[3])

    # Convert lists to numpy arrays and skip the header
    sepal_length = np.array(sepal_length[1:]).astype(np.float64)
    sepal_width = np.array(sepal_width[1:]).astype(np.float64)
    petal_length = np.array(petal_length[1:]).astype(np.float64)
    petal_width = np.array(petal_width[1:]).astype(np.float64)
    
    return np.stack((sepal_length, sepal_width, petal_length, petal_width), axis=1)
