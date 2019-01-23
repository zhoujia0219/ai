from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz, DecisionTreeClassifier
import pandas as pd


def random():
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

    # 4.估计器流程:randomforest
    # 1)实例化估计器类
    estimator = DecisionTreeClassifier()
    # estimator = RandomForestClassifier()

    # # 选择合适的超参数 - 网格搜索
    # param_dict = {"n_estimator": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 12, 30]}
    # estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)

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
    # export_graphviz(estimator, out_file="titanic.dot", feature_names=transfer.get_feature_names())

    # # 结果分析
    # print("在交叉验证中的最好结果: \n", estimator.best_score_)
    # print("最好的参数模型: \n", estimator.best_estimator_)
    # print("每次交叉验证后的验证集准确率结果和训练集准确率结果: \n", estimator.cv_results_)

    return None


if __name__ == '__main__':
    random()