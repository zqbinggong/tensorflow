import math
import tempfile
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags # 用以在命令行执行tf程序时设置参数，在命令行中指定的参数会被tf读取，并转化为flags
flags.DEFINE_string('data_dir', 'mnist-data', 'directory for storing mnist data')
flags.DEFINE_integer('hidden_units', 100, 'number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 1000000, 'number of (global) training steps to perform')
flags.DEFINE_integer('batch_size', 100, 'training batch size')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')

# 是否使用同步计算，以及在同步计算时需要积攒多少个batch的梯度才进行一次参数更新
flags.DEFINE_boolean('sync_replicas', False, '.')
flags.DEFINE_integer('replicas_to_aggregate', None, '.')

# 根据具体的集群定义地址,以及job_name 和 task_index
flags.DEFINE_string('ps_hosts', '192.168.233.201:2222', '.')
flags.DEFINE_string('worker_hosts', '192.168.233.202:2222,192.168.233.203:2222', '.')
flags.DEFINE_string('job_name', None, '.')
flags.DEFINE_integer('task_index', None, '.')

FLAGS = flags.FLAGS # 简化运算
IMAGE_PIXELS = 28

def main(unused_args):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('')
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError(' ')

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # 创建cluster对象，首先需要知道有多少个worker，
    # 创建server用以连接到cluster
    # 如果当前节点是parameter server，则不再进行后续的操作，而是使用server.join等待worker工作
    num_workers = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)
    worker_device = '/job:worker/task:%d/gpu:0' % FLAGS.task_index
    with tf.device(tf.train.replica_device_setter(
        worker_device=worker_device, # 计算资源
        ps_device='/job:ps/cpu:0', # 存储模型参数的资源
        cluster= cluster
    )):
        global_step = tf.Variable(0, naem='global_step', trainable=False)

        # 定义神经网模型
        hid_w, hid_b, sm_w, sm_b, x, y_, hid_lin, hid, y, cross_entropy = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        opt = None

        # 对于同步训练，需要创建同步训练的优化器，本质上时对原有优化器的一个扩展
        if FLAGS.sync_replicas:
            if FLAGS.repicas_to_aggregate is None:
                replicas_to_aggregate = num_workers
            else:
                replicas_to_aggregate = FLAGS.repicas_to_aggregate
            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_workers,
                # replica_id=FLAGS.task_index,
                name='mnist_sync_replicas'
            )

        train_step = opt.minimize(cross_entropy, global_step=global_step)

        # 如果是同步模式并且为主节点，则创建队列执行器，并创建全局参数初始化器
        if FLAGS.sync_replicas and is_chief:
            chief_queue_runner = opt.get_chief_queue_runner()
            init_tokens_op = opt.get_init_tokens_op()

        # 生成本地的参数初始化操作
        # 创建临时的训练目录
        # 创建分布式训练的监督器，管理我们的task参与到分布式训练
        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()
        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir=train_dir,
                                 init_op=init_op,
                                 recovery_wait_secs=1,
                                 global_step=global_step)

        # 设置session参数，allow_soft_placement当某个device不能执行时，是否可以换到其他device执行
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=['/job:ps', '/job:worker/task:%d' % FLAGS.task_index]
        )

        # 如果时主节点，则显示初始化session，其他节点则显示等待主节点的初始化操作
        if is_chief:
            print('')
        else:
            print('')
        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        # 训练过程
        pass

