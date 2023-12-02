import os
import numpy as np
import pandas as pd
import json

'''
data_hanning_path = os.path.join(os.path.dirname(__file__), '..', 'time_hanning_eigenvector')
hanning_vectors_path = os.path.abspath(data_hanning_path)
data_hamming_path = os.path.join(os.path.dirname(__file__), '..', 'time_hamming_eigenvector')
hamming_vectors_path = os.path.abspath(data_hamming_path)


with open(hanning_vectors_path, 'r') as file:
    data_hanning = json.load(file)
with open(hamming_vectors_path,'r') as file:
    data_hamming = json.load(file)
    '''

def x_y_loader( data ):#data是一个嵌套列表，第一维是数字，第二维是说话人，第三维是特征向量
    
    vector_length = len(data[0][0])
    train_data = np.array([[[0.0 for _ in range(vector_length)] for _ in range(12)] for _ in range(10)])#前12个人
    test_data = np.array([[[0.0 for _ in range(vector_length)] for _ in range(5)] for _ in range(10)])#后5个人

    for i in range(10):
        for j in range(12):
            for k in range(vector_length):
                train_data[i][j][k] = data[i][j][k]
        for j in range(5):
            for k in range(vector_length):
                test_data[i][j][k] = data[i][j+12][k]

    X_train = [[0.0 for _ in range(vector_length)] for i in range(120)]
    y_train = []
    X_test =  [[0.0 for _ in range(vector_length)] for i in range(50)]
    y_test = []
    for i in range(10):
        for j in range(12):
            for k in range(vector_length):

                X_train[i*12+j][k] = train_data[i][j][k]
            y_train.append(i)
        for j in range(5):
            for k in range(vector_length):
                X_test[i*5+j][k] = test_data[i][j][k]
            y_test.append(i)
    return X_train, y_train, X_test, y_test