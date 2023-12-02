import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def SVM(X_train, y_train, X_test, y_test):
    # 模型训练，启用概率估计
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    joblib.dump(model, 'SVM_model.pkl')

    # 模型测试
    predictions = model.predict(X_test)

    # 获取测试集上每个类别的概率
    probabilities = model.predict_proba(X_test)

    # 获取每个测试样本的前三个最可能的类别
    top3_classes = np.argsort(-probabilities, axis=1)[:, :3]

    # 计算真实类别是否在前三个最高概率的类别中
    hits = np.sum([y_test[i] in top3_classes[i] for i in range(len(y_test))])

    # 计算 hit@3 指标
    hit_at_3 = hits / len(y_test)

    # 性能评估
    accuracy = accuracy_score(y_test, predictions)



    return predictions, accuracy, hit_at_3
