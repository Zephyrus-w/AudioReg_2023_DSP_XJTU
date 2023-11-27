import os
import numpy as np
from signal_split import all_processed

'''
在拿到all_processed数组后，还需要对其中的数字进行归一化。
这个归一化应该是要消除各个人对同一个数字的大小差异。所以只需要把一个人说话的音强控制到[-1,+1]即可
'''
#遍历每个人
max_amplitude=0
for i in range(17):  # 遍历每个说话者
    for j in range(10):
        for k in range(len(all_processed[i][j])):
            max_amplitude= max(all_processed[i][j][k], max_amplitude)
    # 归一化该说话人的所有数字
    for j in range(10):
        for k in range(len(all_processed[i][j])):
            all_processed[i][j][k]/=max_amplitude

# 确定工作目录
base_dir = "dataset/pre_processed" 
base_path = os.path.join(os.path.dirname(__file__), '..')

# 遍历所有二维数组
for i, two_d_array in enumerate(all_processed):
    # 为每个二维数组创建一个文件夹
    dir_name = os.path.join(base_path,'dataset','pre_processed', f"sample{i+1}")
    os.makedirs(dir_name, exist_ok=True)

    # 为二维数组的每一行创建一个文件
    for j, row in enumerate(two_d_array):
        file_name = os.path.join(dir_name, f"s{i+1}_n{j}")
        with open(file_name, 'w') as file:
            # 将行数据写入文件
            for item in row:
                file.write(f"{item}\n")
