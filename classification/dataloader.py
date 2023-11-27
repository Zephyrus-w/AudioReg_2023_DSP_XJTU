import os
import numpy as np
'''
从time_domain和frequency_domain中读取数据，读出train[data]和[test_data]
'''

vectors = np.array([[[] for _ in range(17)] for _ in range(10)])

for i in range(10):
    data_file_path = os.path.join(os.path.dirname(__file__), '..', 'time_domain','results',f'n{i}.txt')#或者是'frequency_domain'
    # 确保路径是绝对路径
    absolute_data_file_path = os.path.abspath(data_file_path)


    #17=13+4，4是样本后5个13、14、15、16、17
    # 打开文件并按行读取至vectors二维列表
    with open(absolute_data_file_path, 'r') as file:
        for j, line in enumerate(file):
            # 分割每行的字符串，转换为列表
            vector = line.strip().split(',')

            # （可选）转换为数值类型，例如 float
            vector = [float(x) for x in vector]

            # 将转换后的向量添加到三维列表的适当位置
            vectors[i][j] = vector

train_data = np.array([[[] for _ in range(12)] for _ in range(10)])#前12个人
test_data = np.array([[[] for _ in range(5)] for _ in range(10)])#后5个人

for i in range(10):
    for j in range(12):
        train_data[i][j] = vectors[i][j]
    for j in range(5):
        test_data[i][j] = vectors[i][j+12]
