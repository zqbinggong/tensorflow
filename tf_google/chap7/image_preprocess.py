"""
完整地展示图像预处理的各种方式
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def distort_color(image, color_ordering=0):
    """
    给定一张图片，随机调整图像的色彩，之所以随机，是因为亮度、对比度、饱和度和色相的调整顺序会影响处理结果
    故而随机以降低无关因素堆模型的影响
    :param image:
    :param color_ordering:
    :return:
    """
    if color_ordering == 0:
        pass
    pass


pass
q = tf.FIFOQueue(2, 'int32')
# 同Variable， 需要明确调用初始化过程
init = q.enqueue_many()