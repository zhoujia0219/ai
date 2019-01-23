import tensorflow as tf


def linear_regression():
    """
    自实现线性回归
    :return:
    """
    # 1.准备好要训练的数据集
    with tf.variable_scope("prepare_data"):
        x = tf.random_normal(shape=[100, 1], mean=1.5, stddev=1, dtype=tf.float32)
        # 矩阵相乘
        y_true = tf.matmul(x, [[0.8]]) + 0.7

    # 给权重和偏置随机初始值
    with tf.variable_scope("linear_model"):
        W = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]), dtype=tf.float32)
        b = tf.Variable(initial_value=tf.random_normal(shape=[1, 1, ]), dtype=tf.float32)

        y_predict = tf.matmul(x, W) + b

    # 确定损失函数
    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    # 优化损失
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    # 收集变量
    tf.summary.scalar(name="error", tensor=loss)
    tf.summary.histogram(name="weights", values=W)
    tf.summary.histogram(name="bias", values=b)

    # 合并收集的变量
    merge = tf.summary.merge_all()

    # 变量必须要初始化
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)

        # 创建事件文件
        filewriter = tf.summary.FileWriter()
        # 初始化权重和偏重的值
        print("初始化的权重为%f, 偏置为%f" % (W.eval(), b.eval()))

        # 优化损失
        # 多次运行优化器,多次更新权重系数
        for i in range(500):
            sess.run(optimizer)
            print("优化后的权重为%f, 偏置为%f" % (i, loss.eval(), W.eval(), b.eval()))


            summary = sess.run(merge)
            filewriter.add_summary(summary, i)



    return None


if __name__ == '__main__':
    linear_regression()
