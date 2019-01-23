from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba


def text_count_demo():
    """
    对文本进行特征抽取，countvetorizer
    :return: None
    """
    data = [""]
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


def cut_word(text):
    """
    实现分词
    :param text:
    :return:
    """
    text = " ".join(list(jieba.cut(text)))
    return text


def text_chinese_demo3():
    """
    TF-IDF中文文本特征抽取演示
    :return:
    """
    data = ["今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    chinese_list = []
    for sent in data:
        chinese_list.append(cut_word(sent))
    print(chinese_list)

    # 1、实例化一个转换器类
    # --没有设计spare方法
    # transfer = CountVectorizer(sparse=False)
    # 停用词 不处理 stop_words=["", ""]
    transfer = TfidfVectorizer()
    # 2、传入原始数据进行转换, 调用fit_transform
    data = transfer.fit_transform(chinese_list)
    print("文本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names())


if __name__ == '__main__':
    text_chinese_demo3()