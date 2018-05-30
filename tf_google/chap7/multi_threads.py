"""

"""
import tensorflow as tf

##############################################################################
"""
成样例数据
"""


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 模拟海量数据情况下将数据邪图多个文件，其中num_shards给出文件数量,instance_pre_shard 给出每个文件中数据的数量
num_shards = 2
instance_per_shard = 2
# for i in range(num_shards):
#     # 不同文件名最好以类似0000n-of-0000m的后缀区分，其中m表示数据被总的文件个数，
#     # n表示文件编号。这样方便后期以正则表达式获取文件列表
#     filename = ('data/data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
#     writer = tf.python_io.TFRecordWriter(filename)
#     for j in range(instance_per_shard):
#         example = tf.train.Example(features=tf.train.Features(feature={
#             'i': _int64_feature(i),
#             'j': _int64_feature(j)
#         }))
#         writer.write(example.SerializeToString())
#     writer.close()
###############################################################################
###############################################################################
"""
读取样例数据，tf.train.match_filenames_once  tf.train.string_input_producer
"""
files = tf.train.match_filenames_once('data/data.tfrecords-*')
filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64)
    })

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(files))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(4):
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)


