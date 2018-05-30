import tensorflow as tf

input_node = 784
output_node = 10

image_size = 28 # 28 * 28 = 784
num_channels = 1
num_labels = 10

conv1_deep = 32
conv1_size = 5

conv2_deep = 64
conv2_size = 5

fc_size = 512


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1_conv1'):
        conv1_weights = tf.get_variable('weights', [conv1_size, conv1_size, num_channels, conv1_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('biases', [conv1_deep],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 使用边长为5, 深度为32, 步长为1 的过滤器, 且使用全零填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1,1,1,1], padding='SAME')
        relul = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope('layer2_poll1'):
        poll1 = tf.nn.max_pool(relul, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    with tf.variable_scope('layer3_conv2'):
        conv2_weights = tf.get_variable('weihts', [conv2_size, conv2_size, conv1_deep, conv2_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('biases', [conv2_deep],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2 = tf.nn.conv2d(poll1, conv2_weights, strides=[1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope('layer4_poll2'):
        poll2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    poll_shape = poll2.get_shape().as_list()
    nodes = poll_shape[1] * poll_shape[2] * poll_shape[3]
    reshaped = tf.reshape(poll2, [poll_shape[0], nodes])

    with tf.variable_scope('layer5_fc1'):
        fc1_weights = tf.get_variable('weights', [nodes, fc_size],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('biases', [fc_size],
                                     initializer=tf.constant_initializer(0.0))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6_fc2'):
        fc2_weights = tf.get_variable('weights', [fc_size, num_labels],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc2_biases = tf.get_variable('biases', [num_labels],
                                     initializer=tf.constant_initializer(0.0))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit