import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold

# def minmax_demo():
#     """
#     归一化演示
#     :return:
#     """
#     dating = pd.read_csv("dating.txt")
#     print("dating: \n", dating)
#
#     # 1.实例化转换器类
#     transfer = MinMaxScaler()
#     # 2.调用fit_transform进行数据转换
#     data = transfer.fit_transform(dating[["milage", "Liters", "Consumtime"]])
#     print("转换后的数据: \n", data)
#
#     return None

# def standard_demo():
#     """
#     标准化演示
#     :return:
#     """
#     dating = pd.read_csv("dating.txt")
#     print("dating: \n", dating)
#
#     # 1.实例化转换器类
#     transfer = StandardScaler()
#     # 2.调用fit_transform进行数据转换
#     data = transfer.fit_transform(dating[["milage", "Liters", "Consumtime"]])
#     print("转换后的数据: \n", data)
#
#     return None

def variance_demo():
    """
    删除低方差特征-特征选择
    :return:
    """
    # 1.获取数据
    factor = pd.read_csv("factor_returns.csv")
    print("factor: \n", factor.iloc[:, 1:-2])
    data = factor.iloc[:, 1:-2]

    # 2.实例化转换器类
    transfer = VarianceThreshold(threshold=8)

    # 3.fit_transfer转换数据
    data = transfer.fit_transform(data)
    print("删除低方差特征的结果: \n", data)
    print("形状: \n", data.shape)

    return None

if __name__ == '__main__':
    # minmax_demo()
    # standard_demo()
    variance_demo()