import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from dataloader import train_data, test_data

def bayesian(train_data, test_data):
    # 数据预处理
    # 假设 train_data 和 test_data 的形状分别是 [10, 12, 特征数] 和 [10, 5, 特征数]

    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]
    for i in range(10):
        for j in range(12):
            X_train.append(train_data[i][j])
            y_train.append(i)
        for j in range(5):
            X_test.append(test_data[i][j])
            y_test.append(i)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)


    # 模型训练
    model = GaussianNB()
    model.fit(X_train, y_train)

    # 模型测试
    predictions = model.predict(X_test)

    # 性能评估
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    return predictions, accuracy

# 假设 train_data 和 test_data 是预先加载的三维数组
# train_data = np.load('path_to_train_data.npy')
# test_data = np.load('path_to_test_data.npy')

# 调用函数
# predictions, accuracy = bayesian(train_data, test_data)
