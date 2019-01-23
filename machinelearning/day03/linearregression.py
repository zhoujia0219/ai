from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def my_linear():
    """
    线性回归演示-用正规方程优化-波士顿房价预测
    :return:
    """
    # 1.获取数据集
    boston = load_boston()
    # print(": \n", load)
    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3. 特征工程
    estimator = StandardScaler()

    x_train = estimator.fit_transform(x_train)
    x_test = estimator.fit_transform(x_test)

    # 4.预估器流程
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    # 得出模型
    y_predict = estimator.predict(x_test)
    print("正规方程优化预测值为: \n", y_predict)
    print("正规方程优化结果,模型参数中回归系数为: \n", estimator.coef_)
    print("模型参数中偏置为: \n", estimator.intercept_)


    # 5.模型评估
    # 回归性能评估-均方误差
    error = mean_squared_error(y_test, y_predict)
    print("正规方程均方误差为: \n", error)

    return None


def my_linear2():
    """
    线性回归演示-用梯度下降优化-波士顿房价预测
    :return:
    """
    # 1.获取数据集
    boston = load_boston()
    # print(": \n")
    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3. 特征工程
    estimator = StandardScaler()

    x_train = estimator.fit_transform(x_train)
    x_test = estimator.fit_transform(x_test)

    # 4.预估器流程
    # 改变默认学习率 默认的0.01最优
    estimator = SGDRegressor(learning_rate='constant', eta0=0.001)
    estimator.fit(x_train, y_train)
    # 得出模型
    y_predict = estimator.predict(x_test)
    print("梯度下降预测值为: \n", y_predict)
    print("梯度下降优化结果为,模型参数中回归系数为: \n", estimator.coef_)
    print("模型参数中偏置为: \n", estimator.intercept_)

    # 5.模型评估
    # 回归性能评估-均方误差
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降均方误差为: \n", error)

    return None

if __name__ == '__main__':
    my_linear()
    my_linear2()