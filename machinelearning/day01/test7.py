from sklearn.feature_selection import VarianceThreshold
import pandas as pd


def variance_demo():
    """
    特征选择演示-删除低方差特征
    :return:
    """
    data = pd.read_csv("factor_returns.csv")
    # 选择哪几个特征
    data = data.iloc[:, 1:-2]

    # 1.实例化转换器 threshold方差临界值 小于临界值则删除
    transfer = VarianceThreshold(threshold=8)

    # 2.数据转换
    data = transfer.fit_transform(data)
    print("转换后的数据: \n", data)
    print("转换后的形状: \n", data.shape)

    return None


if __name__ == '__main__':
    variance_demo()