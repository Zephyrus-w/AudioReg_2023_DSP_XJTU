from bayesian import bayesian
from fisher import fisher_discriminant
from decision_tree import decision_tree_classifier
from SVM import SVM
from dataloader import x_y_loader

file_path = 'time_domain_vectors.csv'
X_train, y_train, X_test, y_test = x_y_loader( file_path )

def classify(X_train, y_train, X_test, y_test, algorithm):

    if algorithm=='1':#朴素贝叶斯
        predictions, accuracy, hit_3=bayesian(X_train, y_train, X_test, y_test)
    if algorithm=='2':#Fisher线性判别
        predictions, accuracy, hit_3 =fisher_discriminant(X_train, y_train, X_test, y_test)
    if algorithm=='3':#决策树
        predictions, accuracy, hit_3=decision_tree_classifier(X_train, y_train, X_test, y_test)
    if algorithm=='4':#支持向量机
        predictions, accuracy, hit_3=SVM(X_train, y_train, X_test, y_test)

    return predictions, accuracy, hit_3


algorithm = input("请输入方法选择\n朴素贝叶斯：1\nFisher线性判别：2\n决策树：3\n支持向量机：4\n")
print("正在进行分类计算，请稍后...\n")
print("分类结果为：\n")
predictions, accuracy, hit_3= classify(X_train, y_train, X_test, y_test, algorithm)
print("正确结果为：\n")
print(y_test)
print("\n")
print("实际结果为：\n")
print(predictions)
print("\n")
print("正确率为：\n")
print(accuracy)
print("hit@3为：\n")
print(hit_3)

