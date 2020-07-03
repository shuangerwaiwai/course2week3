import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time

np.random.seed(1)

# y_hat = tf.constant(36, name="y_hat")
# y = tf.constant(39, name="y")
#
# loss = tf.Variable((y-y_hat)**2, name="loss")
# init = tf.global_variables_initializer()

# with tf.Session() as session:
#     session.run(init)
#     print(session.run(loss))

# a = tf.constant(2)
# b = tf.constant(10)
# c = tf.multiply(a, b)
#sess = tf.Session()
# print(sess.run(c))

# x = tf.placeholder(tf.int64, name="x")
# print(sess.run(2*x, feed_dict={x:3}))
# sess.close()

def linear_function():
    np.random.seed(1)

    X = np.random.randn(3,1)
    W = np.random.randn(4,3)
    b = np.random.randn(4,1)

    Y = tf.add(tf.matmul(W, X), b)

    sess = tf.Session()
    result = sess.run(Y)

    sess.close()

    return result

#print("result = " + str(linear_function()))

def sigmoid(z):
    x = tf.placeholder(tf.float32, name="x")

    sigmoid = tf.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x:z})

    return  result
#print("sigmoid(0)=" + str(sigmoid(0)))
#print("sigmoid(12)=" + str(sigmoid(12)))

def one_hot_matrix(lables, C):
    C = tf.constant(C, name="C")

    one_hot_matrix = tf.one_hot(indices=lables, depth=C, axis=0)

    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)

    sess.close()
    return one_hot

lables = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(lables, C=4)
# print(str(one_hot))

def ones(shape):
    ones = tf.ones(shape)
    sess = tf.Session()
    ones = sess.run(ones)
    sess.close()
    return ones

#print("ones=" + str(ones([3])))

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()

index=11
plt.imshow(X_train_orig[index])
#plt.show()
#print("Y=" + str(np.squeeze(Y_train_orig[:, index])))

#扁平化
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

#归一化
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

#转换为独热矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig, 6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig, 6)

# print(str(X_train.shape[1]))
# print(str(X_test.shape[1]))
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    return X, Y

# X, Y = create_placeholders(12288, 6)
# print(X)
# print(Y)

#初始化参数
def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12,25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12,1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6,12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6,1], initializer=tf.zeros_initializer())

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }

    return parameters

# tf.reset_default_graph()
#
# with tf.Session() as sess:
#     parameters = initialize_parameters()
#     print("W1:" + str(parameters["W1"]))
#     print("b1:" + str(parameters["b1"]))

def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

# tf.reset_default_graph()
# with tf.Session() as sess:
#     X, Y = create_placeholders(12288, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     print("Z3:" + str(Z3))

def compute_cost(Z3, Y):
    logits = tf.transpose(Z3) #转置
    lables = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=lables))
    return cost

# tf.reset_default_graph()
# with tf.Session() as sess:
#     X, Y = create_placeholders(12288, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3,Y)
#     print("cost:" + str(cost))

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs=1500, minibatch_size=32, print_cost=True, is_plot=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m/minibatch_size)
            seed = seed+1
            minibatches = tf_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_x, minibatche_y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_x, Y:minibatche_y})
                epoch_cost = epoch_cost + minibatch_cost/num_minibatches

            if epoch % 5 == 0:
                costs.append(epoch_cost)
                if print_cost and epoch % 100 == 0:
                    print(str(epoch) + " th, cost:" + str(epoch_cost))

        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate=" + str(learning_rate))
            plt.show()

        parameters = sess.run(parameters)
        print("参数已保存到session")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("训练集的准确率:" , accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集的准确率：", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

stat_time = time.clock()

parameters = model(X_train, Y_train, X_test, Y_test)

ebd_time = time.clock()

print("CPU执行时间：" + str(ebd_time - stat_time) + "秒")