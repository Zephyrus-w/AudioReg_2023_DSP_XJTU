import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from dataloader import train_data, test_data

def fisher_discriminant(train_data, test_data):
    # 数据预处理
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in range(10):  # 10个数字
        for j in range(12):  # 训练集：12个人
            X_train.append(train_data[i][j])
            y_train.append(i)
        for j in range(5):  # 测试集：5个人
            X_test.append(test_data[i][j])
            y_test.append(i)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 模型训练
    model = LinearDiscriminantAnalysis()
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
# predictions, accuracy = fisher_discriminant(train_data, test_data)
