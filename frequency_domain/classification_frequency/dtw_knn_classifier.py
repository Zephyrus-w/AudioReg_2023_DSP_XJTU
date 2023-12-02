from dataloader import x_y_loader
from sklearn.neighbors import KNeighborsClassifier
from dataloader import dtw_distance
from sklearn.metrics import accuracy_score
import numpy as np


def dtw_knn_predict(X_train, y_train, X_test, k):
    predictions = []
    top3_classes = []
    for test_sample in X_test:
        # 计算每个测试样本与所有训练样本之间的 DTW 距离
        distances = [dtw_distance(np.array(test_sample), np.array(train_sample)) for train_sample in X_train]

        # 获取距离最近的 k 个样本的索引
        k_nearest_neighbors = np.argsort(distances)[:k]

        # 获取这些样本的标签
        k_nearest_labels = [y_train[i] for i in k_nearest_neighbors]

        # 预测标签是出现次数最多的标签
        predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        predictions.append(predicted_label)
        # 获取前三个最可能的类别
        top3 = [y_train[i] for i in np.argsort(distances)[:3]]
        top3_classes.append(top3)

    return predictions, top3_classes        



def dtw_knn_classifier(data, k):
    X, y = x_y_loader(data)
    X_train, y_train = X[:120], y[:120]
    X_test, y_test = X[120:], y[120:]
    X_train = [[float(item) for item in row] for row in X_train]
    y_train = [float(item) for item in y_train]
    X_test = [[float(item) for item in row] for row in X_test]
    y_test = [float(item) for item in y_test]

    predictions, top3_classes = dtw_knn_predict(X_train, y_train, X_test, k)
    accuracy = accuracy_score(y_test, predictions)

    hit_at_3 = np.mean([y_test[i] in top3_classes[i] for i in range(len(y_test))])

    return predictions, accuracy, hit_at_3