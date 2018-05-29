import sys
sys.path.append('/home/zq/document/tensorflow/models/tutorials/image/cifar10')
import os.path
import re
import time
import numpy as np
import tensorflow as tf
import cifar10

batch_size = 128
max_steps = 1000000
num_gpus = 4

def tower_loss(scope):
    """
    :param scope: 我们需要为每个GPU生成单独的结构完全一致的网络，由scope标识
    :return:
    """
    images, labels = cifar10.distored_inputs()
    logits = cifar10.inference(images)
    _ = cifar10.loss(logits, labels)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss


def average_gradients(tower_grads):
    """
    讲不同GPU计算得到的梯度进行concat,并进行平均
    :param tower_grads: 梯度的双层列表，基本元素为二元组（梯度，变量），
                        形如[[(grad0_gpu0, var0_gpu0),(grad1_gpu1, var1_gpu1)...],[(grad0_gpu1, var0_gpu1)]...]
    :return:
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads): # ((gradi_gpu0, vari_gpu0),(gradi_gpu1, vari_gpu1)...)
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0) # 方便之后的concat
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        average_grads.append((grad, v))
    return average_grads


def train():
    """

    """
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_intializer(0), trainable=False)
        num_batches_per_epoch = cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
        decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
        lr = tf.train.exponential_decay(cifar10.INITIAL_LEAARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        cifar10.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        opt = tf.train.GradientDescentOptimizer(lr)

        tower_grads = []
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
                    loss = tower_loss(scope)
                    tf.get_variable_scope().reuse_variables()
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
        grads = average_gradients(tower_grads)
        # 更新模型参数
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        pass
