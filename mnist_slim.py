#!/usr/bin/env python
# An MNIST interactive demo

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib import slim
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt


# import cv2


def build_network(images):
    with tf.name_scope('LeNet'):
        net = tf.reshape(images, [-1, 28, 28, 1])
        net = slim.conv2d(net, 16, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        net = slim.conv2d(net, 32, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        net = slim.flatten(net)
        net = slim.fully_connected(net, 128, scope='fc3')
        embeddings = net  # So we can see the outputs in tensorboard
        predictions = slim.fully_connected(net, 10, activation_fn=None,
                                           scope='fc4')
    return predictions, embeddings


def train_net(basedir):
    print('basedir = ', basedir)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    shutil.rmtree(basedir, ignore_errors=True)
    os.mkdir(basedir)

    # The training data:
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    # Build the network, based on LeNet architecture:
    predictions, embeddings = build_network(x)

    # Define our loss function:
    compute_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions))
    compute_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(predictions, axis=1), tf.argmax(y, axis=1)),
                tf.float32))
    # Define the optimizer:
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(
        compute_loss)

    # Show progress with TensorBoard:
    tf.summary.scalar('Loss', compute_loss)
    tf.summary.scalar('Accuracy', compute_accuracy)
    summaries = tf.summary.merge_all()

    # Now start optimizing:
    with tf.Session() as sess:
        # Optimize:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(basedir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(basedir + '/test', sess.graph)

        losses = []
        accuracies = []
        print("%10s | %10s | %5s%% | %5s%%" % (
        'Samples', 'Loss', 'Train', 'Test'))
        for i in range(1, 500):
            batchx, batchy = mnist.train.next_batch(32)
            _, loss, acc, summ = sess.run([train_step, compute_loss,
                                           compute_accuracy, summaries],
                                          feed_dict={x: batchx, y: batchy})
            losses.append(loss)
            accuracies.append(acc)
            train_writer.add_summary(summ, i)

            if i % 20 == 0:
                tloss, tacc, summ = sess.run(
                    [compute_loss, compute_accuracy, summaries],
                    feed_dict={x: mnist.test.images,
                               y: mnist.test.labels})
                print("%10d | %10.6f | %5.2f%% | %5.2f%%" % (
                i * len(batchy), np.mean(losses),
                100.0 * np.mean(accuracies),
                100.0 * tacc))
                losses = []
                accuracies = []
                test_writer.add_summary(summ, i)

        # Save the resulting model:
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, basedir + '/mnist_checkpoint', global_step=i)


def train_or_load_model(basedir):
    latest_checkpoint = tf.train.latest_checkpoint(basedir)

    if not latest_checkpoint:
        train_net(basedir)
        latest_checkpoint = tf.train.latest_checkpoint(basedir)
        print('Now re-run to play with the model')
        exit()

    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None, 784])
    # Build the network, based on LeNet architecture:
    predictions, embeddings = build_network(x)

    saver = tf.train.Saver()
    saver.restore(sess, latest_checkpoint)

    def evaluate_model(images):
        return sess.run(predictions, feed_dict={x: images})

    return evaluate_model


def show_images_and_labels(images, labels):
    plt.figure()
    plt.clf()
    n = images.shape[0]

    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(float(n) / ncols))

    for i in range(n):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(images[i].reshape([28, 28]), cmap='gray')
        plt.axis('off')
        plt.text(0, 27, str(labels[i]), color=(0.2, 1.0, 0.2))

    plt.show()


def examine_model(model):
    # show test-set errors:
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    test_predictions = model(mnist.test.images)
    test_predictions = np.argmax(test_predictions, axis=1)
    acc = test_predictions == np.argmax(mnist.test.labels, axis=1)
    mean_acc = np.mean(acc)
    print("Test accuracy is: %f%%" % (100 * mean_acc))
    mistakes = np.where(~acc)[0]
    print('Found %d mistakes' % len(mistakes))
    mistakes = mistakes[:100]  # Keep max 100
    show_images_and_labels(mnist.test.images[mistakes],
                           test_predictions[mistakes])


if __name__ == "__main__":
    basedir = 'mnist_models'

    model = train_or_load_model(basedir)

    examine_model(model)

    # interactive_demo(model)
