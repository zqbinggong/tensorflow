"""
对weights进行L2正则化;对图片进行翻转、随机裁剪等数据增强，制造了更多的样本;
在每个卷积0最大池化层后面使用了LRN(local response normalization)层，增强了模型的泛化能力
LRN： 对局部神经元的活动创建竞争环境， 使得其中相应较大的值变得更大并抑制其他反馈较小神经元，
这对ReLU这种没有上限边界的激活函数很有用，因为它会从附近的多个卷积核的响应中挑选比较大的反馈，但是不适合sigmoid
这种有固定边界并且能抑制过大值的激活函数（可以理解二阶导数在零附近最大）

数据： CIFAR-10
3000个batch，每个batch包含128个样本，正确率约为0.73
100k个batch，并使用 dacay learning rate， 可达到0.86
"""
import sys
sys.path.append('/home/zq/document/tensorflow/models/tutorials/image/cifar10')

import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time
import math

max_steps = 3000
batch_size = 128
data_dir = '/home/zq/document/tensorflow/tfin_action/chap5-cnn/cifar10_data/cifar-10-batches-bin'


def variable_with_weights_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


# cifar10.maybe_download_and_extract()
# distorted_inputs 会对数据进行增强
images_train, labels_train = cifar10_input.distorted_inputs(data_dir, batch_size)
# 对于test，不需要太多操作，但需要裁剪图片正中间24×24的块并进行数据标准化操作（增强操作中就包含了裁剪）
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

# 由于batch_size在之后定义网络结构时被用到了，所以此处的样本数需要被预先设定，而不能设置成None
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

"""
创建第一个卷积层
卷积层： 尺寸5×5， 步长1, 深度3, 个数64(即卷积之后的深度), 第一层的weight不进行L2正则化，激活函数： ReLU
池化层： 尺寸3×3, 步长2×2, 此处步长和size不一致，可以增加数据的丰富性
LRN：
"""
weight1 = variable_with_weights_loss(shape=[5, 5, 3, 64], stddev=0.05, wl=0.0)
biases1 = tf.Variable(tf.constant(0.0, shape=[64]))
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, biases1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

"""
创建第二层，相比于第一层，只改变了池化层和LRN层的顺序
"""
weight2 = variable_with_weights_loss(shape=[5, 5, 64, 64], stddev=0.05, wl=0.0)
biases2 = tf.Variable(tf.constant(0.1, shape=[64]))
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, biases2))
# 改变了池化层和LRN层的顺序
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

"""
创建第三层， 全连接层， 384个节点
首先flatten卷积层的输出
权重考虑L2惩罚
"""
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weights_loss([dim, 384], stddev=0.04, wl=0.004)
biases3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + biases3)

"""
创建第四层， 全连接层， 192个节点
"""
weight4 = variable_with_weights_loss([384, 192], stddev=0.04, wl=0.004)
biases4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + biases4)

"""
创建最后一层， 全连接层， 192个节点
weight 的标准差设为上一层节点数的导数，并且不计入L2
输出的softmax操作放在了loss里，这里不需要进行softmax
"""
weight5 = variable_with_weights_loss([192, 10], stddev=1/192.0, wl=0.0)
biases5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.nn.relu(tf.matmul(local4, weight5) + biases5)


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


loss = loss(logits, label_holder)

train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 启动distorted_inputs中的线程， 共16个
tf.train.start_queue_runners()

for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss],
                             feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = 'step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch'
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

num_examples = 10000
num_iter = int(math.ceil(num_examples/batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1
precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)

