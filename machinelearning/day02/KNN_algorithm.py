from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def knn_iris():
    """
    KNN算法对鸢尾花分类
    :return:
    """
    # 1.获取数据集
    iris = load_iris()

    # 2.划分数据集 随机划分 加随机数种子可以固定
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=8)

    # 3.特征工程'标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.估计器流程:KNN
    # 1)实例化估计器类 K值就是n_neighbors
    estimator = KNeighborsClassifier(n_neighbors=3)
    # 2)调用fit
    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 方法1:比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("比对真实值和预测值: \n", y_test == y_predict)
    # 方法2: 直接计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为: \n", score)

    return None


if __name__ == '__main__':
    # 1.对鸢尾花分类
    knn_iris()