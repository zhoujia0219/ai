from sklearn.decomposition import PCA

def pca_demo():
    """
    主成分分析演示
    :return:
    """
    data = [[2,8,4,5], [6,3,0,8], [5,4,9,1]]
    # 1.实例化转换器类
    # 小数 保留信息的百分比
    # transfer = PCA(n_components=0.9)
    # 整数 制定降维到几个特征
    transfer = PCA(n_components=2)
    # 2调用 fit_transform
    data = transfer.fit_transform(data)
    print("转换后的数据: \n", data)

    return None


if __name__ == '__main__':
    pca_demo()