import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = 2


def tensorflow_demo():
    pass


def graph_demo():
    default_g = tf.get_default_graph
    print("获取默认图: \n", default_g)

    new_g = tf.Graph()

    with new_g.as_default():
        new_a = tf.constant(value=5)

    print("new_a: \n", new_a.graph)


def session_demo():
    a = tf.constant(10)
    b = tf.constant(20)
    c = tf.add(a, b)

    # 会话创建
    # 1.实例化一个对象
    sess = tf.Session()
    print("c的值为:\n", sess.run(c))
    # 2.用完会话的资源后对资源回收
    sess.close()

    return None


if __name__ == '__main__':
    # tensorflow_demo()
    # graph_demo()
    session_demo()
