from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def knn_iris():
    """
    KNN算法对鸢尾花进行分类
    :return:
    """
    #1.获取数据集
    iris = load_iris()

    #2.划分数据集 随机划分 加随机数种子可以固定
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=8)

    #3.特征工程:标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    #4.估计器流程:KNN
    # 1)实例化估计器类
    estimator = KNeighborsClassifier(n_neighbors=3)

    #增加网格搜索和交叉验证
    param_dict = {"n_neighbors": [1, 3, 5, 7]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)

    # 2)调用fit
    estimator.fit(x_train, y_train)

    #5.模型评估
    #方法1.比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("比对真实值和预测值: \n", y_test == y_predict)

    #方法2.直接计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为: \n", score)

    # 结果分析
    print("在交叉验证中的最好结果: \n", estimator.best_score_)
    print("最好的参数模型: \n", estimator.best_estimator_)
    print("每次交叉验证后的验证集准确率结果和训练集准确率结果: \n", estimator.cv_results_)

    return None

if __name__ == '__main__':
    knn_iris()
















