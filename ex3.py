import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time


def affine_layer_trunc(input, dim_out, act=tf.nn.relu):
    input_shape = input.get_shape().as_list()
    input_dim = input_shape[-1]
    weights_shape = [input_dim, dim_out]

    weights_init = tf.truncated_normal(weights_shape, stddev=0.1)
    bias_init = tf.constant(0.1, shape=[dim_out])

    weights = tf.Variable(weights_init)
    biases = tf.Variable(bias_init)
    preactivate = tf.matmul(input, weights) + biases

    activations = act(preactivate, name='activation')
    return activations


def affine_layer_uniform(input, dim_out, act=tf.nn.relu):
    input_shape = input.get_shape().as_list()
    input_dim = input_shape[-1]
    weights_shape = [input_dim, dim_out]

    weights_init = tf.random_uniform(weights_shape)
    bias_init = tf.constant(0.1, shape=[dim_out])

    weights = tf.Variable(weights_init)
    biases = tf.Variable(bias_init)
    preactivate = tf.matmul(input, weights) + biases

    activations = act(preactivate)
    return activations


def conv_layer_trunc(input, output_channels, ksize, stride, act=tf.nn.relu):
    input_shape = input.get_shape().as_list()
    input_channels = input_shape[-1]

    initial_weight = tf.truncated_normal(
        [ksize, ksize, input_channels, output_channels])
    initial_biases = tf.zeros([output_channels])

    weights = tf.Variable(initial_weight)
    biases = tf.Variable(initial_biases)
    c = tf.nn.conv2d(input, weights, strides=[1, stride, stride, 1],
                     padding='SAME') + biases

    return act(c)


def conv_layer_uniform(input, output_channels, ksize, stride, act=tf.nn.relu):
    input_shape = input.get_shape().as_list()
    input_channels = input_shape[-1]

    initial_weight = tf.random_uniform(
        [ksize, ksize, input_channels, output_channels])
    initial_biases = tf.ones([output_channels])

    weights = tf.Variable(initial_weight)
    biases = tf.Variable(initial_biases)
    c = tf.nn.conv2d(input, weights, strides=[1, stride, stride, 1],
                     padding='SAME') + biases

    return act(c)


def create_architucture_default(x, hidden_act=tf.nn.relu, initialize=True):
    images = tf.reshape(x, [-1, 28, 28, 1])

    if initialize:
        conv1 = conv_layer_trunc(images, 32, ksize=5, stride=1, act=hidden_act)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')
        conv2 = conv_layer_trunc(pool1, 64, ksize=3, stride=1, act=hidden_act)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

    else:
        conv1 = conv_layer_uniform(images, 32, ksize=5, stride=1,
                                   act=hidden_act)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')
        conv2 = conv_layer_uniform(pool1, 64, ksize=3, stride=1, act=hidden_act)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

    shape = pool2.get_shape().as_list()
    shape = [-1, np.product(shape[1:])]
    flat = tf.reshape(pool2, shape)
    aff = affine_layer_trunc(flat, 1024, act=hidden_act)
    predict = affine_layer_trunc(aff, 10, act=tf.identity)
    return predict


def create_architucture_diff_arch(x, hidden_act=tf.nn.relu):
    images = tf.reshape(x, [-1, 28, 28, 1])

    conv1 = conv_layer_trunc(images, 16, ksize=3, stride=1, act=hidden_act)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    conv11 = conv_layer_trunc(pool1, 32, ksize=3, stride=1, act=hidden_act)
    pool11 = tf.nn.max_pool(conv11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME')
    conv2 = conv_layer_trunc(pool11, 64, ksize=3, stride=1, act=hidden_act)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

    shape = pool2.get_shape().as_list()
    shape = [-1, np.product(shape[1:])]
    flat = tf.reshape(pool2, shape)
    aff = affine_layer_trunc(flat, 1024, act=hidden_act)
    predict = affine_layer_trunc(aff, 10, act=tf.identity)
    return predict


def test_batch_size():
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    eta = 0.001
    prediction = create_architucture_default(x)
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)
    cross_entropy = tf.reduce_mean(diff)
    train_step = tf.train.GradientDescentOptimizer(eta).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    batch_options = [30, 300]
    all_train_errors = []
    all_test_errors = []
    for s in batch_options:
        start = time.time()
        test, train = train_net(s, train_step, accuracy, x, y)
        duration = time.time() - start
        all_train_errors.append(train)
        all_test_errors.append(test)
        print("time took for batch size ", str(s), " is:", str(duration))

    plot_errors(all_test_errors, all_train_errors, "batch size", batch_options)


def test_learning_rates():
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    batch_size = 100
    prediction = create_architucture_default(x)
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)
    cross_entropy = tf.reduce_mean(diff)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    eta_options = [1, 0.1, 0.01, 0.001]
    all_train_errors = []
    all_test_errors = []
    for learn_rate in eta_options:
        print('start: %s' % str(learn_rate))
        train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(
            cross_entropy)
        test, train = train_net(batch_size, train_step, accuracy, x, y)
        all_train_errors.append(train)
        all_test_errors.append(test)

    plot_errors(all_test_errors, all_train_errors, "learning rate", eta_options)


def test_learning_algorithms():
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    batch_size = 100
    eta = 0.001
    prediction = create_architucture_default(x)
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)
    cross_entropy = tf.reduce_mean(diff)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_options = [tf.train.AdamOptimizer(eta).minimize(cross_entropy),
                     tf.train.GradientDescentOptimizer(eta).minimize(
                         cross_entropy)]
    all_train_errors = []
    all_test_errors = []
    for train_step in train_options:
        test, train = train_net(batch_size, train_step, accuracy, x, y)
        all_train_errors.append(train)
        all_test_errors.append(test)

    plot_errors(all_test_errors, all_train_errors, "algorithm",
                ["AdamOptimizer", "GradientDescentOptimizer"])


def test_activation():
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    batch_size = 100
    eta = 0.001

    all_train_errors = []
    all_test_errors = []
    hidden_activation_functions = [tf.nn.tanh, tf.nn.relu]
    for act in hidden_activation_functions:
        prediction = create_architucture_default(x, act)
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                       logits=prediction)
        cross_entropy = tf.reduce_mean(diff)
        train_step = tf.train.GradientDescentOptimizer(eta).minimize(
            cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        test, train = train_net(batch_size, train_step, accuracy, x, y)
        all_train_errors.append(train)
        all_test_errors.append(test)

    plot_errors(all_test_errors, all_train_errors, "activation",
                ["TanH", "ReLU"])


def test_initialization():
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    batch_size = 100
    eta = 0.001

    all_train_errors = []
    all_test_errors = []
    initialization_options = [True, False]
    act = tf.nn.relu
    for opt in initialization_options:
        prediction = create_architucture_default(x, act, opt)
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                       logits=prediction)
        cross_entropy = tf.reduce_mean(diff)
        train_step = tf.train.GradientDescentOptimizer(eta).minimize(
            cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        test, train = train_net(batch_size, train_step, accuracy, x, y)
        all_train_errors.append(train)
        all_test_errors.append(test)

    plot_errors(all_test_errors, all_train_errors, "initialization",
                ["truncated_normal", "random_uniform"])


def test_architecture():
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    batch_size = 100
    eta = 0.001

    all_train_errors = []
    all_test_errors = []
    architecture_options = [create_architucture_default,
                            create_architucture_diff_arch]
    act = tf.nn.relu

    for opt in architecture_options:
        prediction = opt(x, act)
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                       logits=prediction)
        cross_entropy = tf.reduce_mean(diff)
        train_step = tf.train.GradientDescentOptimizer(eta).minimize(
            cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        test, train = train_net(batch_size, train_step, accuracy, x, y)
        all_train_errors.append(train)
        all_test_errors.append(test)

    plot_errors(all_test_errors, all_train_errors, "architecture",
                ["2 conv", "3 conv"])


def plot_errors(test_errors, train_errors, param_desc, param_values):
    x_axis = np.arange(0, len(test_errors[0]))

    for i in range(len(param_values)):
        cur_parm = str(param_values[i])
        plt.plot(x_axis, test_errors[i], label=str(
            'test with %s %s' % (param_desc, cur_parm)))
        plt.plot(x_axis, train_errors[i], label=str(
            'train with %s %s' % (param_desc, cur_parm)))

    plt.title("detection, testing the change using different %s" % param_desc)

    plt.ylabel("detection")
    plt.xlabel("iterations")

    plt.legend(loc='lower right')

    plt.savefig(str("success_rates_%s.png" % param_desc), format='png')
    plt.show()


def train_net(batch_size, train_step, accuracy, x, y):
    DEFAULT_ITERATION = 200
    DEFAULT_TEST_FREQ = 10

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    acc_list = []

    train_error = []
    test_error = []

    for i in range(DEFAULT_ITERATION):
        xs, ys = mnist.train.next_batch(batch_size)
        ys = ys.astype(np.float32)
        acc, _ = sess.run([accuracy, train_step], feed_dict={x: xs, y: ys})
        acc_list.append(acc)

        if i % DEFAULT_TEST_FREQ == 0:
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                     y: mnist.test.labels})
            train_error.append(np.mean(acc_list))
            test_error.append(test_acc)
            acc_list = []

    sess.close()

    return test_error, train_error


if __name__ == "__main__":
    mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

    test_batch_size()
    test_learning_rates()
    test_learning_algorithms()
    test_activation()
    test_initialization()
    test_architecture()
