import os
import numpy as np
import pandas as pd
'''
从time_domaim_vectors和frequency_domain_vectors中读取数据，读出train[data]和[test_data]
'''

def convert_specific_csv_to_3d_array(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path, header=None)
    three_d_list = [[[[] for _ in range(2)] for _ in range(17)] for _ in range(10)]
    for i in range(10):
        for j in range(17):
            character = df.iloc[i][j]
            character = character.strip('[]')
            character = character.split(', ')
            for k in range(2):
                three_d_list[i][j][k] = character[k]
                three_d_list[i][j][k] = float(three_d_list[i][j][k])

    # Convert the string representation of lists in the dataframe to actual lists
    # Here, we parse each string as a list of two float values
    # Convert the dataframe to a numpy array
    array=np.array(three_d_list)

    # Check if the array has the correct number of elements (10x17x2)
    if array.shape == (10, 17, 2):
        return array
    else:
        return "Dimension mismatch. Expected a CSV file with 10 rows and 17 columns."
def x_y_loader( file_path ):
    current_directory = os.path.dirname(__file__)
    filepath= os.path.join(current_directory, file_path )# 确保路径是绝对路径
    vectors = convert_specific_csv_to_3d_array(filepath)
    

    train_data = np.array([[[0.0,0.0] for _ in range(12)] for _ in range(10)])#前12个人
    test_data = np.array([[[0.0,0.0] for _ in range(5)] for _ in range(10)])#后5个人

    for i in range(10):
        for j in range(12):
            for k in range(2):
                train_data[i][j][k] = vectors[i][j][k]
        for j in range(5):
            for k in range(2):
                test_data[i][j][k] = vectors[i][j+12][k]

    X_train = [[0.0, 0.0] for i in range(120)]
    y_train = []
    X_test =  [[0.0, 0.0] for i in range(50)]
    y_test = []
    for i in range(10):
        for j in range(12):
            X_train[i*12+j][0] = train_data[i][j][0]
            X_train[i*12+j][1] = train_data[i][j][1]
            y_train.append(i)
        for j in range(5):
            X_test[i*5+j][0] = test_data[i][j][0]
            X_test[i*5+j][1] = test_data[i][j][1]
            y_test.append(i)
    return X_train, y_train, X_test, y_test