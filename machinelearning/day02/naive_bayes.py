from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def news_nb():
    """
    naive_bayes算法对新闻进行分类
    :return: 
    """
    # 1.获取数据集
    news = fetch_20newsgroups(subset="all")
    print("新闻数据: \n", news.data)

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

    # 3.特征工程:文本特征抽取
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.估计器流程:naive_bayes
    # 1)实例化估计器类
    estimator = MultinomialNB(alpha=1.0)

    # 2)调用fit
    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 方法1.比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("比对真实值和预测值: \n", y_test == y_predict)

    # 方法2.直接计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率: \n", score)

    return None


if __name__ == '__main__':
    news_nb()