
import tensorflow as tf

'''
创建文件列表， 并通过文件列表创建输入文件队列
'''
files = ['output.tfrecords']
filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })
image_raw = features['image_raw']
label = features['label']
pixels = features['pixels']

decoded_image = tf.decode_raw(image_raw, tf.uint8)
decoded_image.set_shape(pixels, pixels)
pass
