# adam 优化?
import numpy as np
import tensorflow as tf
import gym
# 创建一个CartPole-v0的环境
env = gym.make('CartPole-v0')
env.reset()

'''
# 进行随机策略看看效果，作为baseline
## 初始化环境


random_episodes = 0
reward_sum = 0
while random_episodes < 10:
    env.render() # 将问题的图像渲染出来
    observation, reward, done, _ = env.step(np.random.randint(0, 2))
    reward_sum += reward
    if done: # 任务结束
        random_episodes += 1
        print('reward for this eplison was :', reward_sum)
        reward_sum = 0
        env.reset()
    '''

# 策略网络，使用一个带有一个隐含层的MLP
## 超参数的设置
H = 50 # 隐藏层节点数
batch_size = 25
learning_rate = 0.1
D = 4 # 环境信息的维度
gamma = 0.99 # 即discount的比例

## 具体的网络结构, 输入：observation， 输出：action（概率）
observations = tf.placeholder(tf.float32, [None, D], name='input_x')
W1 = tf.get_variable('W1', shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable('W2', [H, 1], initializer=tf.contrib.layers.xavier_initializer())
score = tf.nn.relu(tf.matmul(layer1, W2))
prop = tf.nn.sigmoid(score)

adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name='batch_grad1')
W2Grad = tf.placeholder(tf.float32, name='batch_grad2')
batch_grad = [W1Grad, W2Grad]
tvars = tf.trainable_variables()
updateGrads = adam.apply_gradients(zip(batch_grad, tvars))


def discount_rewards(r):
    """
    估算每一个Action对应的潜在discount_r
    :param r: action序列对应的实际reward列表
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


input_y = tf.placeholder(tf.float32, [None, 1], 'input_y') # 人工设置的虚拟label
advantages = tf.placeholder(tf.float32, name='reward_signal')
loglike = tf.log(input_y * (input_y - prop) + (1 - input_y) * (input_y + prop)) # prop为 label为0的action的概率
loss = -tf.reduce_mean(loglike * advantages)

newGrads = tf.gradients(loss, tvars)

xs, ys, drs = [], [], []
reward_sum = 0
episode_number = 1
total_episodes = 10000

with tf.Session() as sess:
    rendering = False
    init = tf.global_variables_initializer()
    sess.run(init)

    observation = env.reset()

    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = 0

    while episode_number <= total_episodes:
        if reward_sum / batch_size > 100 or rendering:
            env.render()
            rendering = True

        x = np.reshape(observation, [1, D])
        tfprob = sess.run(prop, feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0

        xs.append(x)
        y = 1 - action
        ys.append(y)

        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward)

        if done:
            episode_number += 1
            # 产生了一个训练样本
            # 变成列向量以供网络使用
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs, ys, drs = [], [], []

            discount_epr = discount_rewards(epr)
            # 分布稳定，有利于训练的稳定
            discount_epr -= np.mean(discount_epr)
            discount_epr /= np.std(discount_epr)

            tGrad = sess.run(newGrads, feed_dict={
                observations: epx,
                input_y: epy,
                advantages: discount_epr
            })
            for ix, grad in enumerate(tGrad):
                # 加入这个样本的梯度，需要累加batch_size个样本的梯度
                gradBuffer[ix] += grad

            if episode_number % batch_size == 0: # 注意epis_number初始值设为1而不是零
                sess.run([updateGrads], feed_dict={W1Grad: gradBuffer[0],
                                                   W2Grad: gradBuffer[1]})

                for ix, _ in enumerate(gradBuffer):
                    gradBuffer[ix] = 0

                print('average reward for episode % d : %f .' % (episode_number, reward_sum / batch_size))

                if reward_sum / batch_size > 200:
                    print('task solved in', episode_number, 'episodes')
                    break

                reward_sum = 0
            observation = env.reset()
