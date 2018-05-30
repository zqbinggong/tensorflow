"""
使用RNN实现语言模型
"""
import sys
sys.path.append('/home/zq/document/tensorflow/models/tutorials/rnn/ptb')
import numpy as np
import tensorflow as tf
from builtins import print
import reader

data_path = 'data/sinple-examples/data'
hidden_size = 200
num_layers = 2
vocab_size = 10000
learning_rate = 1.0
train_batch_size = 20
train_num_step = 35

# 测试时不需要进行截断，因而可以直接看成超长序列
eval_batch_size = 1
eval_num_step = 1

num_epochs = 2
keep_prob = 0.5
max_grad_norm = 5


class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义输入和预期输出
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])

        # 定义使用LSTM结构为循环结构且使用dropout的深层循环神经网络
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

        # 初始化最初的状态，也就是全零的向量
        self.initial_state = cell.zero_state(batch_size, tf.int32)
        # 将单词ID转成向量
        embedding = tf.get_variable('embedding', [vocab_size, hidden_size])
        # 输入向量 batch_size * num_stpes * hidden_size
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        if is_training: tf.nn.dropout(inputs, keep_prob)

        # 定义输出列表，即先将不同时刻LSTM的输出收集起来，再通过一个全连接层得到最终的输出
        outputs = []
        state = self.initial_state
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        # 改变输出序列的shape，先将其展开为[batch, hidden_size*num_steps],再reshape策成[batch*num_steps, hidden_size]
        # -1 means flatten
        output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])

        # 使用一个全连接层将output变成最后的预测结果。 该结果的维度是vocab_size，经过softmax表示下一个位置是不同单词的概率
        weight = tf.get_variable('weights', [hidden_size, vocab_size])
        bias = tf.get_variable('bias', [vocab_size])
        logits = tf.matmul(output, weight) + bias
        # 使用sequence_loss_by_example来计算序列的交叉熵的和
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example (
            [logits],
            [tf.reshape(self.targets, [-1])]
            [tf.ones(self.batch_size * self.num_steps)]
        )




