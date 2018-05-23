import collections
import tensorflow as tf
slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block"""


def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    """
    这个函数实现了1.slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
                2.subsample(net, factor=stride) 这两个过程，
    之所以要定义新的padding，是因为过程1会导致inputs填充的全零行数和列数都是kernel_size - 1, 而'SAMW"模式填充的全零的数量
        与inputs的size是有关的，故不能直接使用
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=scope)
    else:
        pad_total = kernel_size - 1
        # 以尽量均匀为首要和上少下多、坐少右多为辅助的原则进行分配
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding='VALID', scope=scope)


# 使用@slim.add_arg_scope封装函数，从而可以具备自动参数赋值功能
# （slim.fun中的fun都被这样装饰过，这样就能通过arg_scope自动参数赋值）
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i+1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit # 解包
                    net = block.unit_fn(net, depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            # 将net取名sc.name并加入到outputs_collections中
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    return net


def resnet_arg_scope(is_trianing=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': is_trianing,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.le_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
            return arg_sc


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
    """
    每一层都使用BM，对输入进行preactivation,而不是在卷积进行激活函数处理
    slim.util.last_dimension获取最后一个维度，即输出通道数， 参数mini_rank可以限定最少维度
    使用ReLU进行reactivate
    对于shortcut， 如果输入输出通道数相同，则直接进行降采样，使得空间尺寸相同，如果不同，则使用1x1同步长的卷积核进行处理
    residual： 3层 1x1xdepth_bottleneck/1 3x3xdepth_bottleneck/stride  1x1xdepth/1
    注意最后一层既没有正则项也没有激活函数
    """
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        # 这里貌似意思是经过卷积处理改变空间尺寸前，需要进行BM
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shprtcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None, scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = slim.conv2d_same(residual, depth_bottleneck, 3, stride=stride, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv1')

        output = shortcut + residual
    return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v2(inputs, blocks, num_classes=None, global_pool=True,
              include_root_block=True, reuse=tf.AUTO_REUSE, scope=None):
    """
    :param global_pool: 是否加上最后的一层全局平均池化，使用tf.reduce_mean实现全局平均池化效率更高
    :param include_root_block: 是否加上最前买你通常使用的7x7卷积和最大池化
    """
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense], outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                    net = slim.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = stack_blocks_dense(net, blocks)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True)
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')
    return net, end_points


def resnet_v2_50(inputs, num_classes=None, global_pool=True, reuse=tf.AUTO_REUSE, scope='resnet_v2_50'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)



