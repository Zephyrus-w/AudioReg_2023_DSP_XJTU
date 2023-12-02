from dtw_knn_classifier import dtw_knn_classifier
import json
import os
import numpy as np
import joblib

window = input("Please input your ideal window: hanning or hamming")
if window == 'hanning':
    data_path = os.path.join(os.path.dirname(__file__), '..', 'frequency_eigenvector_delta.json')
    abosulute_data_path = os.path.abspath(data_path)
elif window == 'hamming':
    data_path = os.path.join(os.path.dirname(__file__), '..', 'frequency_eigenvector.json')
    abosulute_data_path = os.path.abspath(data_path)


with open(abosulute_data_path, 'r') as file:
    data = json.load(file)

algorithm = input("请输入您想要的算法：\n KNN：1\n")
if algorithm == '1':
    k = int(input("请输入您想要的K值：\n"))
    predictions, accuracy, hit_at_3 = dtw_knn_classifier(data, k)



print(f'准确率：{accuracy}\n')
print(f'hit@3值：{hit_at_3}\n')
print(f'预测结果：{predictions}\n')   
