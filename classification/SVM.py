import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from dataloader import train_data, test_data

def SVM(train_data, test_data):
    # 数据预处理
    X_train = []
    y_train = []
    X_test = []
    y_test = []

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
    model = SVC()
    model.fit(X_train, y_train)

    # 模型测试
    predictions = model.predict(X_test)

    # 性能评估
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    return predictions, accuracy

# 使用函数
# predictions, accuracy = svm_classifier(train_data, test_data)
