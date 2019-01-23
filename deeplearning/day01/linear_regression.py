import tensorflow as tf


def linear_regression():
    """
    自实现线性回归
    :return:
    """
    # 1.准备好要训练的数据集
    x = tf.random_normal(shape=[100, 1], mean=1.5, stddev=1, dtype=tf.float32)
    # 矩阵相乘
    y_true = tf.matmul(x, [[0.8]]) + 0.7

    # 给权重和偏置随机初始值
    W = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]), dtype=tf.float32)
    b = tf.Variable(initial_value=tf.random_normal(shape=[1, 1, ]), dtype=tf.float32)

    y_predict = tf.matmul(x, W) + b

    # 确定损失函数
    loss = tf.reduce_mean(tf.square(y_true - y_predict))

    # 优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    # 变量需要初始化
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)

        # 初始化权重和偏重的值
        print("初始化的权重为%f, 偏置为%f" % (W.eval(), b.eval()))

        # 优化损失
        # 多次运行优化器,多次更新权重系数
        for i in range(500):
            sess.run(optimizer)
            print("优化后的权重为%f, 偏置为%f" % (W.eval(), b.eval()))

    return None


if __name__ == '__main__':
    linear_regression()
