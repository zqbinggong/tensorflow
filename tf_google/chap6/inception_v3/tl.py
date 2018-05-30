import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# inception-v3瓶颈层的节点数,以及瓶颈层结果的名称; 在训练模型时，可以通过tensor.name 获取获取张量名称
bottleneck_tensor_size = 2048
bottleneck_tensor_name = 'pool_3/_reshape:0'

# 图像输入张量对应的名称
jpeg_data_tensor_name = 'DecodeJpeg/contents:0'

model_dir = 'model'
model_file = 'tensorflow_inception_graph.pb'

# 因为一个数据可能被使用多次，所以可以将原始图像通过inception-v3模型得到的特征向量保存在文件中
cache_dir = 'temp/bottleneck'

input_data = 'flower_photos'

validation_percentage = 10
test_percentage = 10

learning_rate = 0.01
steps = 4000
batch = 100


def creat_image_lists(test_percentage, validation_percentage):
    """
    切分数据
    :param test_percentage:
    :param validation_percentage:
    :return: 字典结构result, key为类名，value为图片名称
    """
    result = {}
    sub_dirs = [x[0] for x in os.walk(input_data)]
    '''
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', "JPEG"]
        file_list = []
        # dir_name : the name of flower
        dir_name = os.path.basename(sub_dir) # obtain the tail of sub_dir, None if sub_dir end with '/' or '\'
        for extension in extensions:
            file_glob = os.path.join(input_data, dir_name, '*.'+extension)
            file_list.append(glob.glob(file_glob))

        if not file_list: continue
        label_name = dir_name.lower()
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (test_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
        }
    '''
    sub_dirs.pop(0)
    for sub_dir in sub_dirs:
        dir_name = os.path.basename(sub_dir)
        extensions = ['jpg', 'jpeg', 'JPG', "JPEG"]
        file_list = []
        for extension in extensions:
            file_glob = os.path.join(sub_dir, '*.'+extension)
            file_list.append(glob.glob(file_glob))

        if not file_list: continue
        label_name = os.path.basename(sub_dir).lower()
        training_images = []
        testing_images = []
        validation_images = []
        for file_names in file_list:
            for file_name in file_names:
                chance = np.random.randint(100)
                file_name = os.path.basename(file_name)
                if chance < validation_percentage:
                    validation_images.append(file_name)
                elif chance < (test_percentage + validation_percentage):
                    testing_images.append(file_name)
                else:
                    training_images.append(file_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
        }

    return result


def get_image_path(image_lists, image_dir, label_name, index, category):
    """
    :param image_lists: 所有图片信息, 即上面函数返回的result
    :param image_dir: 根目录，存放图片数据的根目录和存放图片特征向量的根目录地址不同
    :param label_name: 类名
    :param index: 需要获取的图片的编号
    :param category: 需要获取的图片属于train，test还是validation
    :return: 图片地址
    """
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, category):
    """
    这里返回的是特征向量的路径
    :param image_lists:
    :param label_name:
    :param index:
    :param category:
    :return:
    """
    return get_image_path(image_lists, cache_dir, label_name, index, category)


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    """
    使用训练好的inceptio-v3 处理一张图片，得到这个图片的特征向量
    :param sess:
    :param image_data:
    :param image_data_tensor:
    :param bottleneck_tensor:
    :return:
    """
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    """
    获取图片的特征向量，如果特征向量还没被计算过，则计算出并保存到文件
    :param sess:
    :param image_lists:
    :param label_name:
    :param index:
    :param category:
    :param jpeg_data_tensor:
    :param bottleneck_tensor:
    :return:
    """
    # 获取图片对应的特征向量文件的路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(cache_dir, sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path) # makedies 可以建立多层目录，如data/train/image

    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)

    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, input_data, label_name, index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read() # 'r': utf-8 encoding, 'rb': not utf-8 encoding
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many,
                                  category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, category,
                                              jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for label_index, label_name in enumerate(image_lists.keys()):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category,
                                                  jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main(_):
    image_lists = creat_image_lists(test_percentage, validation_percentage)
    n_classes = len(image_lists.keys())

    # 读取保存在GraphDefProtocolBuffer中模型
    with gfile.FastGFile(os.path.join(model_dir, model_file), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # 相当于返回了一个入口和一个出口
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[bottleneck_tensor_name, jpeg_data_tensor_name])

    # 定义新的神经网络的输入，即图片经过insepti-v3前向传播获得的特征向量
    bottleneck_input = tf.placeholder(tf.float32, [None, bottleneck_tensor_size],
                                      name='BottleneckInputPlaceholder')

    # 定义新的标准答案输入
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes],
                                        name='GroundTruthInput')

    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([bottleneck_tensor_size, n_classes], stddev=0.1))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    # define loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean)

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(steps):
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, batch, 'training', jpeg_data_tensor, bottleneck_tensor)
            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            if i % 1000 == 0 or i + 1 == steps:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, batch, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step,
                                               feed_dict={bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('step %d : validation accuracy on random sampled %d examples = %.1f%%' %
                      (i, batch, validation_accuracy*100))

        test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes,
                                                                   jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step,
                                 feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('final test accuracy = %.1f%%' % (test_accuracy*100))


if __name__ == '__main__':
    tf.app.run()
