# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
#
# def datasets_demo():
#     """
#     对鸢尾花数据集的演示
#     :return:
#     """
#     # 1.获取鸢尾花数据集
#     iris = load_iris()
#     # 返回结果是bunch 继承于字典
#     print("鸢尾花数据集的返回值 \n", iris)
#     print("鸢尾花的特征值 \n", iris.data)
#     print("鸢尾花的特征值 \n", iris["data"])
#     print("鸢尾花的目标值 \n", iris.target)
#     print("鸢尾花的描述 \n", iris.DESCR)
#     print("鸢尾花的特征值名字 \n", iris.feature_names)
#     print("鸢尾花的目标值名字 \n", iris.target_names)
#
#     # 机器学习一般的数据集会划分为训练和测试2个部分
#     # 训练数据用于训练构建模型  测试数据在模型检验时用于评估模型是否有效  占比在20%~30%之间
#     # 2.对鸢尾花数据集进行划分
#     x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=8)
#     print("x_train: \n", x_train.shape)
#     # 随机数种子
#     x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target, random_state=6)
#     x_train2, x_test2, y_train2, y_test2 = train_test_split(iris.data, iris.target, random_state=6)
#     print("如果随机数种子不一致", x_train == x_train1)
#     print("如果随机数种子一致: \n", x_train1 == x_train2)
#
#     return None
#
# if __name__ == '__main__':
#     datasets_demo()


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def iris():
    """
    对鸢尾花数据集的演示
    :return:
    """
    # 1.获取数据
    iris = load_iris()

    print("鸢尾花数据集的返回值类型: \n", type(iris))
    print("鸢尾花数据集的特征值: \n", iris.data)
    print("鸢尾花数据集的目标值: \n", iris.target)
    print("鸢尾花数据集的特征值名字: \n", iris.feature_names)
    print("鸢尾花数据集的目标值名字: \n", iris.target_names)
    print("鸢尾花数据集的描述: \n", iris.DESCR)

    # 2.划分数据集(训练-测试)
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=8)
    print("x_train: \n", x_train.shape)

    # 随机数种子
    x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target, random_state=6)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(iris.data, iris.target, random_state=6)
    print("如果随机数种子不一致", x_train == x_train1)
    print("如果随机数种子一致: \n", x_train1 == x_train2)

    return None

if __name__ == '__main__':
    iris()