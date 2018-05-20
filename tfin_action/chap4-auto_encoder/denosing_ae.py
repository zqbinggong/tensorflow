"""
使用tensorflow实现去噪自编码器
数据集为mnist
需要用到一种新的参数初始化方法 xavier initialization
"""
import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(fan_in, fan_out, constant = 1):
    """
    使用均匀或者高斯分布来获得满足要求的初始化权重，即均值为零， 方差为 2/(fan_in + fan_out)
    """
    low = -constant * np.sqrt(6.0 / (fan_in - fan_out))
    high = constant * np.sqrt(6.0 / (fan_in - fan_out))
    return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high, dtype=tf.float32)


class AdditiveGaussianNoiseEncoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        """
        注意此处只使用了一个隐藏层
        :param n_input: 输入变量数
        :param n_hidden: 隐藏层节点数
        :param transfer_function: 激活函数
        :param optimizer:
        :param scale: 高斯噪声系数
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # define input and output
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + self.scale * tf.random_normal([n_input]),
                                                   self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # define loss and create session
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        """
        这里只用隐藏层有激活函数，而输出层不需要激活函数，因而直接初始化为零即可
        :return:
        """
        all_weights = {'w1': tf.Variable(xavier_init(self.n_input, self.n_hidden)),
                       'b1': tf.Variable(tf.zeros([self.n_hidden])),
                       'w2': tf.Variable(tf.zeros([self.n_hidden, self.n_input])),
                       'b2': tf.Variable(tf.zeros(self.n_input))}
        return all_weights

    def partial_fit(self, X):
        costs, _ = self.sess.run([self.cost, self.optimizer],
                                 feed_dict={self.x : X, self.scale: self.training_scale})
        return costs

    def calc_total_cost(self, X):
        """
        用于测试集
        """
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruction(self, X):
        """
        与上面的函数进行比较，这里是完整运行，即包括transform和generate两部分
        """
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    def get_biases(self):
        return self.sess.run(self.weights['b1'])


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    """
    属于不放回抽样，可以提供提高数据的利用效率
    """
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]

mnist = input_data.read_data_sets('../../tf_google/chap5/data/mnist', one_hot=True)
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

autoencoder = AdditiveGaussianNoiseEncoder(n_input = 784,
                                           n_hidden = 200,
                                           transfer_function=tf.nn.softplus,
                                           optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                           scale=0.01)
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(xs)
        avg_cost += cost / n_samples
    if epoch % display_step == 0:
        print('epoch: %04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
print('test mean cost: ', str(autoencoder.calc_total_cost(X_test) / int(mnist.test.num_examples)))
'''
epoch: 0001 cost= 159.025459126
epoch: 0002 cost= 96.678213841
epoch: 0003 cost= 86.959709100
epoch: 0004 cost= 76.531243683
epoch: 0005 cost= 75.599880513
epoch: 0006 cost= 75.599254461
epoch: 0007 cost= 72.216389959
epoch: 0008 cost= 64.154048122
epoch: 0009 cost= 64.153571871
epoch: 0010 cost= 61.699715336
epoch: 0011 cost= 67.749097940
epoch: 0012 cost= 62.446931521
epoch: 0013 cost= 61.422716184
epoch: 0014 cost= 63.700597230
epoch: 0015 cost= 64.452473801
epoch: 0016 cost= 65.229329834
epoch: 0017 cost= 63.019468577
epoch: 0018 cost= 59.295072199
epoch: 0019 cost= 59.881019922
epoch: 0020 cost= 62.479350630
test mean cost:  68.2696375
'''