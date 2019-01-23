from sklearn.feature_extraction.text import CountVectorizer

def text_count_demo():
    """
    对文本进行特征抽取，countvetorizer
    :return: None
    """
    data = ["life is short,i like like python", "life is too long,i dislike python"]
    # 1、实例化一个转换器类
    # --没有设计spare方法
    # transfer = CountVectorizer(sparse=False)
    # 停用词 不处理 stop_words=["", ""]
    transfer = CountVectorizer(stop_words=["is"])
    # 2、传入原始数据进行转换, 调用fit_transform
    data = transfer.fit_transform(data)
    print("文本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names())

    return None

if __name__ == '__main__':
    text_count_demo()