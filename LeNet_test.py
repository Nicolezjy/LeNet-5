from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

FLAGS = None

data_dir = 'MNIST_data/'
mnist = input_data.read_data_sets(data_dir, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

learning_rate = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])

regularizer = tf.contrib.layers.l2_regularizer(0.001)

#model
with tf.variable_scope('conv1'):
    w_conv1 = tf.get_variable('weight', [5, 5, 1, 6], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_conv1 = tf.get_variable('bias', [6], initializer=tf.constant_initializer(0.1))
    conv1 = tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='VALID')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b_conv1))

with tf.variable_scope('pool'):
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.variable_scope('conv2'):
    w_conv2 = tf.get_variable('weight', [5, 5, 6, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_conv2 = tf.get_variable('bias', [16], initializer=tf.constant_initializer(0.1))
    conv2 = tf.nn.conv2d(pool1, w_conv2, strides=[1, 1, 1, 1], padding='VALID')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2))

with tf.variable_scope('pool2'):
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

pool_shape = pool2.get_shape().as_list()
nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

reshaped = tf.reshape(pool2, [-1, nodes])

with tf.variable_scope('fc1'):
    w_fc1 = tf.get_variable('weight', [nodes, 120], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_fc1 = tf.get_variable('bias', [120], initializer=tf.constant_initializer(0.1))
    tf.add_to_collection('losses', regularizer(w_fc1))
    fc1 = tf.nn.relu(tf.matmul(reshaped, w_fc1) + b_fc1)
    tf.nn.dropout(fc1, 0.5)

with tf.variable_scope('fc2'):
    w_fc2 = tf.get_variable('weight', [120, 84], initializer=tf.truncated_normal_initializer(stddev=0.1))
    tf.add_to_collection('losses', regularizer(w_fc2))
    b_fc2 = tf.get_variable('bias', [84], initializer=tf.constant_initializer(0.1))
    fc2 = tf.nn.relu(tf.matmul(fc1, w_fc2) + b_fc2)
    tf.nn.dropout(fc2, 0.5)

with tf.variable_scope('fc3'):
    w_fc3 = tf.get_variable('weight', [84, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
    tf.add_to_collection('losses', regularizer(w_fc2))
    b_fc3 = tf.get_variable('bias', [10], initializer=tf.constant_initializer(0.1))
    y = tf.matmul(fc2, w_fc3) + b_fc3

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.name_scope('losses'):
    losses = cross_entropy + 0.00001 * tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('losses', losses)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(losses)
merged = tf.summary.merge_all()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver(max_to_keep=1)

saver.restore(sess, "model/1")

test_loss, test_acc, batch_num = 0, 0, 0
err, acc = sess.run([losses, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
test_loss += err;
test_acc += acc;
print("test loss:", test_loss / 1)
print("test acc:", test_acc / 1)

