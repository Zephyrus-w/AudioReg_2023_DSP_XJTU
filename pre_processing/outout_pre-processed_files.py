import os
from signal_split import all_processed


# 确定工作目录
base_dir = "dataset/pre_processed" 
base_path = os.path.join(os.path.dirname(__file__), '..')

# 遍历所有二维数组
for i, two_d_array in enumerate(all_processed):
    # 为每个二维数组创建一个文件夹
    dir_name = os.path.join(base_path,'dataset','pretrained', f"sample{i+1}")
    os.makedirs(dir_name, exist_ok=True)

    # 为二维数组的每一行创建一个文件
    for j, row in enumerate(two_d_array):
        file_name = os.path.join(dir_name, f"s{i+1}_n{j}")
        with open(file_name, 'w') as file:
            # 将行数据写入文件
            for item in row:
                file.write(f"{item}\n")
