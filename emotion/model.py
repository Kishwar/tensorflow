# These are my hobby project codes developed in python using OpenCV and TensorFlow
# Some of the projects are tested on Mac, Some on Raspberry Pi
# Anyone can use these codes without any permission
#
# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '25.03.18' '21:13'

# imports
import tensorflow as tf
from constants import *

p_keep_conv = tf.placeholder("float")

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))

# WEIGHTS
def layer_weights():

    w1 = init_weights([5, 5, 1, 32])        # layer 1, 32 5x5 filters
    w2 = init_weights([3, 3, 32, 64])       # layer 2, 64 3x3 filters
    w3 = init_weights([3, 3, 64, 64])      # layer 3, 128 3x3 filters
    w4 = init_weights([64 * 12 * 12, 256])   # fully connected layer
    w_o = init_weights([256, NUM_LABELS])   # output layer

    return w1, w2, w3, w4, w_o

# BIASES
def layer_biases():

    b1 = init_biases([32])                 # layer 1 - same as number of filters for this layer
    b2 = init_biases([64])                 # layer 2 - same as number of filters for this layer
    b3 = init_biases([64])                # layer 3 - same as number of filters for this layer
    b4 = init_biases([256])                # layer 4 - same as number of filters for this layer
    b_o = init_biases([NUM_LABELS])        # output layer - same as number of filters for this layer

    return b1, b2, b3, b4, b_o

# MODEL
def model(X, w1, w2, w3, w4, w_o, b1, b2, b3, b4, b_o, p_keep_conv):

    # model will follow the weights and biases we have defined above

    conv1 = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b1)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, p_keep_conv)
    tf.add_to_collection("losses", tf.nn.l2_loss(w1))    # regularization
    tf.add_to_collection("losses", tf.nn.l2_loss(b1))    # regularization

    conv2 = tf.nn.conv2d(conv1, w2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b2)
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, p_keep_conv)
    tf.add_to_collection("losses", tf.nn.l2_loss(w2))    # regularization
    tf.add_to_collection("losses", tf.nn.l2_loss(b2))    # regularization

    conv3 = tf.nn.conv2d(conv2, w3, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, b3)
    conv3 = tf.nn.relu(conv3)
    tf.add_to_collection("losses", tf.nn.l2_loss(w3))    # regularization
    tf.add_to_collection("losses", tf.nn.l2_loss(b3))    # regularization

    FC_layer = tf.reshape(conv3, [-1, 64 * 12 * 12])
    FC_layer = tf.nn.relu(tf.matmul(FC_layer, w4) + b4)

    output_layer = tf.matmul(FC_layer, w_o) + b_o

    return output_layer

def loss(pred, label):
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label))
    tf.summary.scalar('Entropy', cross_entropy_loss)
    reg_losses = tf.add_n(tf.get_collection("losses"))
    tf.summary.scalar('Reg_loss', reg_losses)
    return cross_entropy_loss + REGULARIZATION * reg_losses


def train(loss, step):
    return tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=step)