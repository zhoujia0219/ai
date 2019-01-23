from sklearn.feature_extraction import DictVectorizer


def dict_demo():
    """
    对字典进行特征抽取
    :return:
    """
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]

    # 1.实例化一个转换器类
    transfer = DictVectorizer()
    # 2.传入数据进行转换
    data = transfer.fit_transform(data)
    print("转换后的特征名: \n", transfer.get_feature_names())
    print("转换后的数据: \n", data)

    return None


if __name__ == '__main__':
    dict_demo()