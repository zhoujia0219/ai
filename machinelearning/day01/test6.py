import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def minmax_demo():
    """
    归一化演示
    :return:
    """
    data = pd.read_csv("dating.txt")

    # 1.实例化转换器类
    transfer = MinMaxScaler()
    # 2.转换数据
    data = transfer.fit_transform(data[["milage", "Liters", "Consumtime"]])
    print("转换结果: \n", data)

    return None

def standard_demo():
    """
    标准化演示
    :return:
    """
    data = pd.read_csv("dating.txt")

    # 1.实例化转换器类
    transfer = StandardScaler()
    # 2.转换数据
    data = transfer.fit_transform(data[["milage", "Liters", "Consumtime"]])
    print("转换结果: \n", data)

    return None


if __name__ == '__main__':
    # minmax_demo()
    standard_demo()