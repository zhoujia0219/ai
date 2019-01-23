import pandas as pd
import numpy as np
import tensorflow as tf
import glob

# 流程分析
# 1) 解析csv文件.建立文件名和标签值对应表格

# 2) 读取图片数据
def read_image():

    # 生成文件名列表
    glob.glob("./GenPics/*.jpg")

    # 1.构造文件名队列
    file_queue = tf.train.string_input_producer(file_list)

    # 2.读取与解码
    reader = tf.WholeFileReader()
    filename, value = reader.read(file_queue)

    # 解码
    image = tf.image.decode_jpeg(value)

    # 固定形状
    image.set_shape([20, 80, 3])

    # 3.批处理
    filename_batch, image_batch = tf.train.batch([filename, image], batch_size=100, num_threads=2, capacity=200)

    return filename_batch, image_batch

# 3) 将标签值的字母转换为0~25的数字
def filename2label(filenames, csv_data):

    labels = []

    for filename in filenames:
        a = "".join(list(filter(str.isdigit, str(filename))))
        labels.append(csv_data.loc[int(file_num), "labels"])

    return np.array(labels)

# 4) 建立卷积神经网络模型
def create_variable(shape):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.01))

def creat_cnn_model(x):

    with tf.variable_scope("conv1"):


    return None

if __name__ == '__main__':
    parse_csv()

    filename_batch, image_batch = read_image()

    # 准备数据,定义占位符
    x = tf.placeholder(tf.float32, [None, 20, 80, 3])
    y_true = tf.placeholder(tf.float32, [None, 4*26])

    y_predict = create_cnn_model(x)

    # 开启会话
    with tf.Session() as sess:
        # 开启线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    filenames, images = sess.run([filename_batch, image_batch])

    labels = filename2label(filenames, csv_data)

    labels_onehot = tf.reshape(tf.one_hot(labels, 26), [-1, 4*26]).eval()

    # 回收线程
    coord.request_stop()
    coord.join(threads)
