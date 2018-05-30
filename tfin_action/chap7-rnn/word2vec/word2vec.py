import tensorflow as tf
import collections
import math
import random
import numpy as np
import zipfile

with zipfile.ZipFile('text8.zip') as f:
    words = tf.compat.as_str(f.read(f.namelist()[0])).split() # 共17005207个单词，总计100000000个字母

vocabulary_size = 50000


def build_dataset(words):
    """
    创建vocabulary词汇表
    1. 使用collections.Count统计单词词频，再使用most_common获取前5000个单词作为vocabulary,其他的单词认定为UNknow，编号为0
    2. 使用dict来存储这些单词，因为dict的查询复杂度为O（1）
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    dictionary = dict()
    for word, _ in count: # 自动解包
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0 # 统计不在dict中的单词个数
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
del words # 删除原始列表以节约内存

data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    """

    :param batch_size: 必须是num_skips的整数倍，这样才能保证每个batch都包含了一个词汇对应的所有样本
    :param num_skips: 每个单词生成的样本数 <= 2 * skip_window
    :param skip_window: 单词最远联系的距离
    """
    global data_index # 这样可以在函数内部对data_index即起始位置进行修改
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data_index)
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            # 后期可以优化
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


batch_size = 128
embedding_size = 128 # 单词向量的维度，一般设在550-1000之间
skip_window = 1
num_skips = 2

# 验证数据相关参数
valid_size = 16 # 用来抽取的验证单词数
valid_window = 100 # 验证单词从频数最高的100个单词中抽取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64 # 训练时用来做负样本的噪声单词的数量

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size))
    )
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()

num_steps = 100001

with tf.Session(graph=graph) as sess:
    init.run()
    avg_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        avg_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                avg_loss /= 2000
            print('average loss at step', step, ':', avg_loss)
            avg_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = 'nearest to %s' % valid_word
                for k in range(top_k):
                    closed_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s' % (log_str, closed_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()





