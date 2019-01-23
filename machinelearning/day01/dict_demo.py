from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

# def datasets_demo():
#     """
#     对鸢尾花数据集的演示
#     :return: None
#     """
#     # 1、获取鸢尾花数据集
#     iris = load_iris()
#     print("鸢尾花数据集的返回值：\n", iris)
#     # 返回值是一个继承自字典的Bench
#     print("鸢尾花的特征值:\n", iris["data"])
#     print("鸢尾花的目标值：\n", iris.target)
#     print("鸢尾花特征的名字：\n", iris.feature_names)
#     print("鸢尾花目标值的名字：\n", iris.target_names)
#     print("鸢尾花的描述：\n", iris.DESCR)
#
#     # 2、对鸢尾花数据集进行分割
#     # 训练集的特征值x_train 测试集的特征值x_test 训练集的目标值y_train 测试集的目标值y_test
#     x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.20, random_state=8)
#     print("x_train:\n", x_train.shape)
#     print("y_train:\n", y_train.shape)
#     # 随机数种子
#     x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target, random_state=6)
#     x_train2, x_test2, y_train2, y_test2 = train_test_split(iris.data, iris.target, random_state=6)
#     print("如果随机数种子不一致：\n", x_train == x_train1)
#     print("如果随机数种子一致：\n", x_train1 == x_train2)
#
#     return None

def dict_demo():
    """
    对字典类型的数据进行特征抽取
    :return: None
    """
    data = [{'city': '北京','temperature':100}, {'city': '上海','temperature':60}, {'city': '深圳','temperature':30}]
    # 1、实例化一个转换器类
    transfer = DictVectorizer(sparse=False)
    # 2、调用fit_transform
    data = transfer.fit_transform(data)
    print("返回的结果:\n", data)
    # 打印特征名字
    print("特征名字：\n", transfer.get_feature_names())

    return None

if __name__ == '__main__':
    dict_demo()