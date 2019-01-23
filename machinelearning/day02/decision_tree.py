from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def decision_iris():
    """
    用决策树对鸢尾花进行分类
    :return:
    """
    # 1.获取数据集
    iris = load_iris()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=8)

    # 3.估计器流程:decisiontree
    # 1)实例化估计器类
    estimator = DecisionTreeClassifier(criterion="entropy", max_depth=5)

    # 2)调用fit
    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 方法1.比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("比对真实值和预测值: \n", y_test == y_predict)

    # 方法2.直接计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率: \n", score)

    # 可视化决策树
    export_graphviz(estimator, out_file="tree.dot", feature_names=iris.feature_names)

    return None


if __name__ == '__main__':
    decision_iris()