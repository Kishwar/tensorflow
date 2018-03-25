# These are my hobby project codes developed in python using OpenCV and TensorFlow
# Some of the projects are tested on Mac, Some on Raspberry Pi
# Anyone can use these codes without any permission
#
# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '25.03.18' '15:05'

import tensorflow as tf
from constants import *

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# WEIGHTS
def layer_weights():

    w1 = init_weights([3, 3, 1, 32])        # layer 1, 32 3x3 filters  - input 1 class (grayscale image)
    w2 = init_weights([3, 3, 32, 64])       # layer 2, 64 3x3 filters  - input 32 class / prev layer filters
    w3 = init_weights([3, 3, 64, 128])      # layer 3, 128 3x3 filters  - input 64 class / prev layer filters
    w4 = init_weights([128 * 4 * 4, 625])   # fully connected layer
    w_o = init_weights([625, num_classes])  # output layer

    return w1, w2, w3, w4, w_o

# MODEL
def model(X, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):

    # model will follow the weights we have defined above

    conv1 = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, p_keep_conv)

    conv2 = tf.nn.conv2d(conv1, w2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, p_keep_conv)

    conv3 = tf.nn.conv2d(conv2, w3, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(conv3)

    FC_layer = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    FC_layer = tf.reshape(FC_layer, [-1, w4.get_shape().as_list()[0]])
    FC_layer = tf.nn.dropout(FC_layer, p_keep_conv)

    output_layer = tf.nn.relu(tf.matmul(FC_layer, w4))
    output_layer = tf.nn.dropout(output_layer, p_keep_hidden)

    result = tf.matmul(output_layer, w_o)
    return result