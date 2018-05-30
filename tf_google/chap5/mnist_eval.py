import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

# 每10s加载一次一次最新的模型，并在测试集上测试最新模型的正确率
eval_interval_secs = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.input_node], name='x_input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.output_node], name='y_input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = mnist_inference.inference(x, None)

        correct_preds = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

        variable_avg = tf.train.ExponentialMovingAverage(mnist_train.moving_average_decay)
        variables_to_restore = variable_avg.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        mcp = ''
        flag = True
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.model_save_path)
                if flag or ckpt.model_checkpoint_path != mcp:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('after %s training step(s), validation accuracy = %g' % (global_step, accuracy_score))
                    mcp = ckpt.model_checkpoint_path
                    flag = False
                else:
                    print('no checkpoint file found')
                    time.sleep(eval_interval_secs)


def main(argv=None):
    mnist = input_data.read_data_sets('data/mnist', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()