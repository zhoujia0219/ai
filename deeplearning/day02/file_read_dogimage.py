import tensorflow as tf
import os


def read_dog():
    """
    读取狗图片
    :return:
    """
    # 1.构造图片文件队列
    file_queue = tf.train.string_input_producer(file_list)

    # 2.1) 读取图片数据
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)

    # 2) 解码
    decoded = tf.image.decode_jpeg(value)

    # 3.处理图片数据形状，批处理返回
    # 在进行批处理之前必须要把形状固定下来,有一个None都不行
    # 统一图片形状并缩小
    image_resized = tf.image.resize_images(decoded, [200, 200])

    # 固定形状
    image_resized.set_shape([200, 200, 3])

    image_batch = tf.train.batch([image_resized], batch_size=10, num_threads=1, capacity=10)

    return image_batch


if __name__ == '__main__':
    # 构造文件名列表
    file_name = os.listdir("./dog/")
    print(file_name)

    # 拼接
    file_list = [os.path.join("./dog/", file) for file in file_name]
    print(file_list)

    image_batch = picread(file_list)

    read_dog()

    # 开启会话线程运行
    with tf.Session() as sess:
        # 开启线程
        # 创建线程协调器
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess, coord=coord)
        print(sess.run(image_batch))

        # 回收资源
        coord.request_stop()
        coord.join(threads)
