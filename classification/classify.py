import bayesian
import fisher
import decision_tree
import SVM
from dataloader import train_data, test_data


def classify(train_data, test_data, way):
    if way==1:#朴素贝叶斯
        predictions, accuracy=bayesian(train_data, test_data)
    if way==2:#Fisher线性判别
        predictions, accuracy=fisher(train_data, test_data)
    if way==3:#决策树
        predictions, accuracy=decision_tree(train_data, test_data)
    if way==4:#支持向量机
        predictions, accuracy=SVM(train_data, test_data)

    return predictions, accuracy

y_test=[]
for i in range(10):
    for j in range(5):
        y_test.append(i)


way = input("请输入方法选择\n朴素贝叶斯：1\nFisher线性判别：2\n决策树：3\n支持向量机：4\n")
print("正在进行分类计算，请稍后...\n")
print("分类结果为：\n")
predictions, accuracy = classify(train_data, test_data, way)
print("正确结果为：\n")
print(y_test)
print("\n")
print("实际结果为：\n")
print(predictions)
print("\n")
print("正确率为：\n")
print(accuracy)

