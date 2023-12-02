import os
import numpy as np
import pandas as pd
import json
from scipy.spatial.distance import euclidean
from dtw import dtw
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
import numpy as np

def dtw_distance(x, y):
    manhattan_distance = lambda x, y: np.abs(x - y)
    d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)
    return d


def x_y_loader(data):
    # 假设 data 是一个10x17的列表，每个元素是一个一维特征向量列表
    num_digits = 10
    num_speakers = 17

    X = []  # 特征数据
    y = []  # 标签数据

    for digit in range(num_digits):
        for speaker in range(num_speakers):
            X.append(data[digit][speaker])
            y.append(digit)

    return X, y