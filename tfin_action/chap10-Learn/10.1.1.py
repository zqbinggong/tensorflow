import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn


def my_model(features, target):
    target = tf.one_hot(target, 3, 1, 0)
    # 堆叠神经网络，，每一层分别有10,20,10个隐藏节点
    features = layers.stack(features, layers.fully_connected, [10,20,10])
    prediction, loss = learn.models.logistic_regression_zero_init(features, target)
    train_op = layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad', learning_rate=0.1)
    return {'class': tf.argmax(prediction, 1), 'prop': prediction}, loss, train_op


from sklearn import datasets, cross_validation
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=35)
classifier = learn.Estimator(model_fn=my_model)
classifier.fit(x_train, y_train, steps=700)
predictions = classifier.predict(x_test)

# 此处在fit和evaluate中还有很多其他参数，可以实现更多的自定义逻辑
# 先在_input_fn里建立数据，使用layers模块建立两个特征列--年龄和性别
def _input_fn(num_epochs=None):
    features = {'age': tf.train.limit_epochs(tf.constant([[.8],[.2],[.1]]), num_epochs=num_epochs),
                # 返回tensor num_epochs次， 并raise on 'OutOfRange' error
                'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                            indices=[[0, 0], [0, 1], [2, 0]], dense_shape=[3, 2])
                # dense.shape = dense_shape, dense[tuple(indices[i])] = values[i]
            }
    return features, tf.constant([[1], [0], [0]], dtype=tf.int32) # 特征和label
language_column = tf.contrib.layers.sparse_column_with_hash_bucket(
    'language', hash_bucket_size=20)
feature_columns = [
    tf.contrib.layers.embedding_column(language_column, dimension=1),
    tf.contrib.layers.real_valued_column('age')
]
# 将特征列、每层的隐藏神经元数、标识类别数等传入DNNClassifier里建立模型
classifier = tf.contrib.learn.DNNClassifier(
    n_classes=2,
    feature_columns=feature_columns,# 注意这里feature_columns相当于placeholder，它需要fit的参数input_fn的返回值作为feed
    # weight_column_name=  , 考虑数据带有权重，这时需要将权重也加入到feature_columns
    hidden_units=[3, 3],
    config=tf.contrib.learn.RunConfig(tf_random_seed=1)
)
classifier.fit(input_fn=_input_fn, steps=100)
scores = classifier.evaluate(input_fn=_input_fn, steps=1)

# load the data
iris_trian = tf.contrib.learn.datasets.base.load_csv(filename='', target_dtype=np.int)
iris_test = tf.contrib.learn.datasets.base.load_csv(filename='', target_dtype=np.int)
# define a metrics dict to evaluate the model
validation_metrics = {'accuracy':tf.contrib.metrics.streaming_accuracy,
                      'precision': tf.contrib.metrics.streaming_precision,
                      'recall': tf.contrib.metrics.streaming_recall}

# use the metrics-dict to construct the validation_monitor
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    iris_test.data,
    iris_test.target, # offer the data and target to estimate the model
    every_n_steps=50, # run this monitor every 50 steps
    metrics=validation_metrics,
    early_stopping_metric='loss', # early stopping depending on the 'loss' metric
    early_stopping_metric_minimize=True, # if True, we should minimize the early_stopping_metric
    early_stopping_rounds=200
)
# next, we construct a DNNClassifier,
# which has 3 layers and the number of hidden units of each layer are 10,15,10
# note that there we can assign multiple monitors to monitor different functions
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 15, 10],
    n_classes=3,
    model_dir='',
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=2)
)
classifier.fit(x=iris_trian.data, y=iris_train.target, steps=10000, monitors=[validation_monitor])
accuracy_score = classifier.evaluate(x=iris_test.data, y=iris_test.target)['accuracy']
# corresponding to validation_metrics dict.

tf.create_partitioned_variables()