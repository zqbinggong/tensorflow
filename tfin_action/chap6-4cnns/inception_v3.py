"""
v1： v1 有22层深，（alexnet8层， vggnet19层）;参数少但效果好的原因除了模型层数更深、表达能力更强外，还有两点：
    去除了最后的全连接层，用全局平均池化层（即将图片尺寸变为1×1来代替）
    设计了inception module提高了参数的利用效率
inception module：
    包含4个分支，每个分支都用到了1×1的卷积，来进行低成本的跨通道的特征变换
    4个分支在最后通过一个聚合操作（在输出通道上）合并
    包含3种不同尺寸的卷积核和一个最大化池，增加了网络堆不同尺度的适应性
    可以让网络的深度和宽度高效率的扩充。提高准确率且不至于过拟合
inception net的主要目标是选找最优的稀疏结构单元（即inception module）， 基于Hebbian原理
hebbian原理：神经反射活动的持续性和重复会导致神经元连接稳定性的持久提升，总结即 cells that fire together, wire together
一个好的稀疏结构，应该是符合hebbian原理的， 即我们应该把相关性高的一簇神经元节点连接在一起。对于图片来说，临近区域的相关性高，在同一
    空间位置但不同通道的卷积核的输出结果相关性极高，因此1×1卷积核是极好的选择，当然3×3、5×5相关性野很高，因为也可以使用稍大的来增加
    多样性
inception module中通常1×1卷积核的比列是最高的，但是在整个网络中会有多个堆叠的im， 我们希望靠后的im能够捕捉到更高阶的抽象特征，因此
    靠后的im的卷积的空间集中度应该逐渐降低，这样可以捕捉更大面积的特征，因此越到后面，大尺度的卷积核的比例应该更多
22层中，除了最后一层的输出， 其中间层节点的分类效果也很好。因此在inception net 还用到了辅助分类节点，即将中间某一层的输出用作分类，
    并按一个小的权重0.3加到最终的分类结果中。这样做相当于做了模型融合，也提供了额外的正则化，对于整个网络的训练很有裨益

v2学习了VGGNet，用两个3×3代替5×5并使用BN
batch normalization： bp在用于某层时，会对每一个mini-batch数据的内部进行标准化处理，使输出规范化到N(0,1)的正态分布，减少了
    internal covariance shift（内部神经元分布的改变）
    对于传统的深度神经网络在训练时， 每一层的输入的分布都在变化，导致训练变得困难，我们只能使用一个很小的学习速率来解决这个问题;
    使用了BP之后，就可以使学习效率增大很多倍。同时，需要作出相应的调整以配合BP，（比v1快了14倍，并且模型在收敛时的准确率更高）
        增大学习速率并加快学习衰减速度以适用BP规范化后的数据;
        去除dropout并减轻L2正则（因为BP已经起到正则化的作用）
        去除LRN
        更彻底地地对训练样本进行shuffle，
        减少数据增强过程中对数据的光学畸变，因为BP训练更快（epochs值更小），每个样本被训练的次数更少，因此更真实的样本堆训练更有帮助

v3 主要是两个方面的改造：
    引入了factorization into small convolution, 较少了参数的同时，还增多了激活函数从而增强了非线性变换
    优化了inception module的结构

值得借鉴的思想和Trick：
    1. factorization into small convolutions很有效，可以降低参数量、减轻过拟合、增加网络非线性的表达能力
    2. 卷积网络从输入到输出， 应该让图片尺寸逐渐减小、输出通道数逐渐增加，即让空间简化，将空间信息转化为高阶抽象的特征信息
    3. 用多个分支提取不同抽象程度的高阶特征的思路很有效， 可以丰富网络的表达能力
"""

import tensorflow as tf
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


# 将自动设置默认参数的arg_scope过程封装成函数，方便使用
def inception_v3_arg_scope(weight_decay=0.00004, stddev=0.1, batch_norm_var_collection='moving_vars'):
    batch_norm_params = {'decay': 0.9997,
                         'epsilon': 0.001,
                         'updates_collections': tf.GraphKeys.UPDATE_OPS,
                         'variables_collections': {
                             'beta': None,
                             'gamma': None,
                             'moving_mean': [batch_norm_var_collection],
                             'moving_variance': [batch_norm_var_collection]
                         }}
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params) as sc:
            return sc


def inception_base(inputs, scope=None):
    """
    用于生成卷积部分
    设计inception net的一个重要原则是图派你尺寸是不断缩小的，同时输出通道不断增加;
    每一层卷积，池化或者im组的目的都是将空间结构简化，同时将空间信息转化为更高阶的特征信息，即将空间的维度转为通道的维度
        这一过程同时也使得每次输出的tensor的总size持续下降，减低了计算量
    im一般有4个分支，分支1一般为1x1卷积， 分支2一般为1x1卷积再接分解后的1xn和nx1卷积，
        分支3和分支2类似但是一半更深一些，分支4一般是具有最大池化或者平均池化
        im通常是通过组合比较简简单的特征抽象（分支1,1x1卷积）、比较复杂的特征抽象（分支2和3）和一个简化结构的池化层（分支4），
        一共4种不同程度的特征抽象和变换来有选择的保留不同层次的高阶特征
    :param inputs: 输入图片数据的tensor
    :param scope: 包含了函数默认参数的环境
    :return: net end_points，后者保存了某些关键节点供之后使用,即用来辅助分类的中间节点
    """
    end_points = {}
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        # 先是对图片尺寸进行压缩，并对图片特征进行抽象，共5个卷积，2个最大池
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='VALID'):
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3')
            net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_2x3')
            net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
            net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')
            net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_3x3')

        # 接下类是3个连续的inception module组
        ## 组1 包含了3个结构类似的inception module
        ### im1 4个分支  所有卷积核步长为1,padding为same， 输出tensor：35x35x256
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0c_5x5')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        ### im2, 与之前的唯一不同在于第四分支最后接的是64输出通道的卷积， 输出tensor：35x35x288
            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0c_5x5')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        ### im3, 与im2一模一样 输出tensor：35x35x288
            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0c_5x5')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        ## 组2 包含了5个inception module， 后4个结构非常相似
        ### im1 3个分支  输出tensor：17x17x768
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID', scope='Conv2d_0a_3x3')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2, padding='VALID', scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_0a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)

        ### im2, 4个分支， 输出tensor：17x17x768
            with tf.variable_scope('Mixed_6b'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        ### im3, 与im2相似, 第二个和第三个分支中前几个卷积层的输出通道数不同,但最终输出通道数相同 输出tensor：35x35x288
            with tf.variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        ### im4, 与im3一模一样, 输出tensor：35x35x288
            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        ### im5, 与im4一模一样, 输出tensor：35x35x288, 将Mixed_6e存储与end_points中，作为Auxiliary Classifier辅助模型的分类
            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points['Mixed_6e'] = net

        ## 组3 包含了3个inception module， 后2个结构非常相似
        ### im1 3个分支  输出tensor：8x8x1280
            with tf.variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_3x3')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_0d_3x3')
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_0a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)

        ### im2, 4个分支， 输出tensor：8x8x2048
            with tf.variable_scope('Mixed_7b'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        ### im3, 与im2一模一样
            with tf.variable_scope('Mixed_7c'):
                with tf.variable_scope('Branch0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                with tf.variable_scope('Branch2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                with tf.variable_scope('Branch3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
    return net, end_points


def inception_v3(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax, spatial_squeeze=True, reuse=None, scope='InceptionV3'):
    """
    最后一部分： 全局平均化，softmax和Auxiliary Logits
    :param inputs:
    :param num_classes:
    :param is_training: 只有在训练时才会启用bn和dropout
    :param dropout_keep_prob:
    :param prediction_fn:
    :param spatial_squeeze: 去除维数为1的维度5x5x1-->5x5
    :param reuse:
    :param scope:
    """
    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = inception_base(inputs, scope)
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                aux_logits = end_points['Mixed_6e']

                # 辅助分类部分，经过卷积池化等操作后，size=1x1x1000
                with tf.variable_scope('AuxLogits'):
                    aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID', scope='AvgPool_1a_5x5')
                    aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='Conv2d_1b_1x1')
                    aux_logits = slim.conv2d(aux_logits, 768, [5, 5], weights_initializer=trunc_normal(0.01),
                                             padding='VALID', scope='Conv2d_2a_5x5')
                    aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn=None,
                                             normalizer_fn=None, weights_initializer=trunc_normal(0.01),
                                             scope='Conv2d_2b_1x1')
                    if spatial_squeeze:
                        aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                    end_points['AuxLogits'] = aux_logits

                # 正常分类
                with tf.variable_scope('Logits'):
                    net = slim.avg_pool2d(net, [8, 8], padding='VALID', scope='AvgPool_1a_8x8')
                    net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                    end_points['PreLogits'] = net
                    logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                         normalizer_fn=None, scope='Conv2d_1c_1x1')
                    if spatial_squeeze:
                        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                    end_points['Logits'] = logits
                    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points


"""
性能测试
"""
from datetime import datetime
import time
import math


def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f =/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))

batch_size = 32
height, width = 299, 299
inputs = tf.random_uniform([batch_size, height, width, 3])
with slim.arg_scope(inception_v3_arg_scope()):
    logits, end_points = inception_v3(inputs, is_training=False, reuse=tf.AUTO_REUSE)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
time_tensorflow_run(sess, logits, 'forward')


"""
2018-05-22 18:14:35.874307: step 0, duration = 18.064
2018-05-22 18:17:36.628911: step 10, duration = 18.041
"""


