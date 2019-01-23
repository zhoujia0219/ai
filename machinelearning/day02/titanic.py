from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
import pandas as pd

def titanic():
    """
    用决策树对泰坦尼克号乘客生存预测
    :return:
    """
    # 1.获取数据集
    titanic = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    # 2.基本的数据处理
    x = titanic[["pclass", "age", "sex"]]
    y = titanic["survived"]

    # 处理缺失值
    x["age"].fillna(x["age"].mean(), inplace=True)

    # 把特征转换成字典形式
    x = x.to_dict(orient="records")

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # 3.特征工程:字典特征抽取
    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.估计器流程:决策树
    # 1)实例化估计器类
    estimator = DecisionTreeClassifier(criterion="entropy", max_depth=4)

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
    export_graphviz(estimator, out_file="titanic.dot", feature_names=transfer.get_feature_names())

    return None


if __name__ == '__main__':
    titanic()