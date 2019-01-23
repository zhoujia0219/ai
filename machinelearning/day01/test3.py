from sklearn.feature_extraction.text import CountVectorizer


def text_demo():
    """
    对文本进行特征抽取
    :return:
    """
    data = ["life is short,i like like python", "life is too long,i dislike python"]

    # 1.实例化一个转换器类
    transfer = CountVectorizer(stop_words=["is"])
    # 2.传入数据进行转换
    data = transfer.fit_transform(data)
    print("转换后的特征名: \n", transfer.get_feature_names())
    print("抽取之后的结果: \n", data.toarray())

    return None


if __name__ == '__main__':
    text_demo()