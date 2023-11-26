import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from SALib.analyze import sobol
from SALib.sample import saltelli

from sklearn.decomposition import KernelPCA  # 核主成分分析法数据降维
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split  # k折交叉训练

# 单一分类器
from sklearn.ensemble import GradientBoostingClassifier  # 梯度提升
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.model_selection import cross_val_score  # 交叉验证
from sklearn.naive_bayes import GaussianNB  # 朴素贝叶斯
from sklearn.neighbors import KNeighborsClassifier  # k近邻算法
from sklearn.neural_network import MLPClassifier  # 多层感知机
from sklearn.svm import SVC  # 支持向量机
from sklearn.tree import DecisionTreeClassifier  # 决策树
from xgboost import XGBClassifier  # xgboost

# 自行编写的绘图函数包
import my_plot

# 模型融合
from sklearn.ensemble import StackingClassifier

"""
此文件包含了模型的训练过程，以及集成学习优化模型
"""

random_state1 = 2333
random_state2 = 4666

# 读取输入
data_train = pd.read_csv('./train_filled.csv')
data_test = pd.read_csv('./titanic/test.csv')

# 利用正则表达式取出需要的属性
train_df = data_train.filter(regex='Survived|Age_.*|SibSp_.*|Parch_.*|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

test_df = data_test.filter(regex='Age_.*|SibSp_.*|Parch_.*|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_np = test_df.values
# 获取y
y = train_np[:, -1].astype(int)
# 获取自变量x
x = train_np[:, :-1]
x_test = test_np

x_tr, x_vl, y_tr, y_vl = train_test_split(x, y, random_state=2333, test_size=0.2)

# 设置分类器
models = [("LR", LogisticRegression()), ("NB", GaussianNB()), ("KNN", KNeighborsClassifier()),
          ("SVM", SVC(probability=True)), ("MLP", MLPClassifier()), ("XGB", XGBClassifier())]

# 训练
results = []
names = []

kfold = KFold(n_splits=10, shuffle=True, random_state=2333)  # n_splits=10表示划分10等份

for name, model in models:
    cv_result = cross_val_score(model, x, y, cv=kfold, scoring="accuracy")
    my_plot.plot_roc_auc(model, name, kfold, x_vl, y_vl)
    names.append(name)
    results.append(cv_result)

# 结果
for i in range(len(names)):
    print(names[i], results[i].mean())  # 10次结果的平均值
    my_plot.plot_learning_curve(models[i][1], (names[i]), x, y)  # 绘制各分类器的学习曲线

ax = sns.boxplot(data=results)
ax.set_xticklabels(names)

# ****************数据降维和超参数调优****************

kpca = KernelPCA(n_components=2, kernel='rbf')  # kernel = 'rbf' 核函数为 高斯核函数，n_components:降维后的维数 2维
x_tr_pca = kpca.fit_transform(x_tr)
# fit_transform(trainData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
x_vl_pca = kpca.transform(x_vl)

plt.figure(figsize=(10, 8))
plt.scatter(x_tr_pca[:, 0], x_tr_pca[:, 1], c=y_tr, cmap='plasma')   # c=y_tr 对应着两种颜色，区分点的颜色
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

# classifier = SVC(kernel='rbf')
# classifier.fit(x_tr_pca, y_tr)

# ****************应用集成学习来优化模型****************

# 逻辑回归没有增加多样性的选项
clf1 = LogisticRegression(max_iter=3000, C=0.1, random_state=random_state1, n_jobs=8)

# 增加特征多样性与样本多样性
clf2 = RandomForestClassifier(n_estimators=100, max_features="sqrt", max_samples=0.9, random_state=random_state1, n_jobs=-1)

# 特征多样性，稍微上调特征数量
clf3 = GradientBoostingClassifier(n_estimators=100, max_features=16, random_state=random_state1)

# 增加算法多样性，新增决策树与KNN
clf4 = DecisionTreeClassifier(max_depth=8, random_state=random_state1)
clf5 = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
clf6 = GaussianNB()

# 新增随机多样性，相同的算法更换随机数种子
clf7 = RandomForestClassifier(n_estimators=100, max_features="sqrt", max_samples=0.9, random_state=random_state2, n_jobs=-1)

clf8 = GradientBoostingClassifier(n_estimators=100, max_features=16, random_state=random_state2)

estimators = [("Logistic Regression", clf1), ("RandomForest", clf2), ("GBDT", clf3),
              ("Decision Tree", clf4), ("KNN", clf5), ("Bayes", clf6),
              ("RandomForest2", clf7), ("GBDT2", clf8)]


# 选择单个评估器中分数最高的随机森林作为元学习器
# 也可以尝试其他更简单的学习器
final_estimator = RandomForestClassifier(n_estimators=100,
                                         min_impurity_decrease=0.0025,
                                         random_state=random_state1, n_jobs=-1)

stkclf = StackingClassifier(estimators=estimators,  # level0的7个个体学习器
                            final_estimator=final_estimator,  # level 1个元学习器
                            n_jobs=-1)

rfclf = RandomForestClassifier(n_estimators=10)

stkclf.fit(x_tr, y_tr)
rfclf.fit(x_tr, y_tr)

# 预测
y_stk_pre = stkclf.predict(x_vl)
y_rf_pre = rfclf.predict(x_vl)

print('The Metrics of Stacking is: \n', classification_report(y_vl, y_stk_pre, target_names=['a类', 'b类']))
# my_plot.draw_confusion_matrix(y_vl, y_bag_pre, ['乘员存活', '乘员死亡'])

print('The Metrics of RandFR is: \n', classification_report(y_vl, y_rf_pre, target_names=['a类', 'b类']))
# my_plot.draw_confusion_matrix(y_vl, y_rf_pre, ['乘员存活', '乘员死亡'])



