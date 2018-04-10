# These are my hobby project codes developed in python using OpenCV and TensorFlow
#
# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

# License
# Copyright (c) 2016 Logan Engstrom. Contact me for commercial use (or rather any use that is not academic research)
# (email: engstrom at my university's domain dot edu). Free for research use, as long as proper attribution is given
# and this copyright notice is retained.

__author__ = 'kishwarkumar'
__date__ = '07.04.18' '07:53'

import tensorflow as tf
from constants import *

# random weights generator
def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    # return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

# WEIGHTS
def layer_weights(FilterSize, NFilters, InChannel, transpose=False):
    if not transpose:
        W = init_weights([FilterSize, FilterSize, InChannel, NFilters])
    else:
        W = init_weights([FilterSize, FilterSize, NFilters, InChannel])
    return W


def noiseModel(TImage):

    # Constructs the graph model.
    graph = {}

    X = graph['input'] = TImage
    # -------------------------------------------------------------
    #                       CONVOLUTION BLOCKS                    #
    # -------------------------------------------------------------
    # Convolution block 1
    W1 = layer_weights(9, 32, X.get_shape().as_list()[3])
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = Layer_Norm(L1)
    L1 = graph['conv1']  = tf.nn.relu(L1)

    # Convolution block 2
    W2 = layer_weights(3, 64, L1.get_shape().as_list()[3])
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 2, 2, 1], padding='SAME')
    L2 = Layer_Norm(L2)
    L2 = graph['conv2']  = tf.nn.relu(L2)

    # Convolution block 3
    W3 = layer_weights(3, 128, L2.get_shape().as_list()[3])
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 2, 2, 1], padding='SAME')
    L3 = Layer_Norm(L3)
    L3 = graph['conv3']  = tf.nn.relu(L3)

    # -------------------------------------------------------------
    #                         RESIDUAL BLOCKS                     #
    # -------------------------------------------------------------
    # Residual block 1
    W4 = layer_weights(3, 128, L3.get_shape().as_list()[3])
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = Layer_Norm(L4)
    L4 = tf.nn.relu(L4)

    W5 = layer_weights(3, 128, L4.get_shape().as_list()[3])
    L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
    L5 = Layer_Norm(L5)

    L6 = graph['res1'] = L3 + L5

    # Residual block 2
    W7 = layer_weights(3, 128, L6.get_shape().as_list()[3])
    L7 = tf.nn.conv2d(L6, W7, strides=[1, 1, 1, 1], padding='SAME')
    L7 = Layer_Norm(L7)
    L7 = tf.nn.relu(L7)

    W8 = layer_weights(3, 128, L7.get_shape().as_list()[3])
    L8 = tf.nn.conv2d(L7, W8, strides=[1, 1, 1, 1], padding='SAME')
    L8 = Layer_Norm(L8)

    L9 = graph['res2'] = L6 + L8

    # Residual block 3
    W10 = layer_weights(3, 128, L9.get_shape().as_list()[3])
    L10 = tf.nn.conv2d(L9, W10, strides=[1, 1, 1, 1], padding='SAME')
    L10 = Layer_Norm(L10)
    L10 = tf.nn.relu(L10)

    W11 = layer_weights(3, 128, L10.get_shape().as_list()[3])
    L11 = tf.nn.conv2d(L10, W11, strides=[1, 1, 1, 1], padding='SAME')
    L11 = Layer_Norm(L11)

    L12 = graph['res3'] = L9 + L11

    # Residual block 4
    W13 = layer_weights(3, 128, L12.get_shape().as_list()[3])
    L13 = tf.nn.conv2d(L12, W13, strides=[1, 1, 1, 1], padding='SAME')
    L13 = Layer_Norm(L13)
    L13 = tf.nn.relu(L13)

    W14 = layer_weights(3, 128, L13.get_shape().as_list()[3])
    L14 = tf.nn.conv2d(L13, W14, strides=[1, 1, 1, 1], padding='SAME')
    L14 = Layer_Norm(L14)

    L15 = graph['res4'] = L12 + L14

    # Residual block 5
    W16 = layer_weights(3, 128, L15.get_shape().as_list()[3])
    L16 = tf.nn.conv2d(L15, W16, strides=[1, 1, 1, 1], padding='SAME')
    L16 = Layer_Norm(L16)
    L16 = tf.nn.relu(L16)

    W17 = layer_weights(3, 128, L16.get_shape().as_list()[3])
    L17 = tf.nn.conv2d(L16, W17, strides=[1, 1, 1, 1], padding='SAME')
    L17 = Layer_Norm(L17)

    L18 = graph['res5'] = L15 + L17

    # -------------------------------------------------------------
    #                   CONV TRANSPOSE BLOCKS                     #
    # -------------------------------------------------------------
    # ConT block 1
    W19 = layer_weights(3, 64, L18.get_shape().as_list()[3], True)
    m, n_H, n_W, n_C = L18.get_shape().as_list()
    m, n_H, n_W, n_C = m, int(n_H * 2), int(n_W * 2), n_C

    L19 = tf.nn.conv2d_transpose(L18, W19, tf.stack([m, n_H, n_W, 64]),
                                 [1, 2, 2, 1], padding='SAME')
    L19 = Layer_Norm(L19)
    L19 = tf.nn.relu(L19)

    # ConT block 2
    W20 = layer_weights(3, 32, L19.get_shape().as_list()[3], True)
    m, n_H, n_W, n_C = L19.get_shape().as_list()
    m, n_H, n_W, n_C = m, int(n_H * 2), int(n_W * 2), n_C
    L20 = tf.nn.conv2d_transpose(L19, W20, tf.stack([m, n_H, n_W, 32]),
                                 strides=[1, 2, 2, 1], padding='SAME')
    L20 = Layer_Norm(L20)
    L20 = tf.nn.relu(L20)

    # -------------------------------------------------------------
    #                       CONVOLUTION BLOCKS                    #
    # -------------------------------------------------------------
    # Convolution block 1
    W21 = layer_weights(9, 3, L20.get_shape().as_list()[3])
    L21 = tf.nn.conv2d(L20, W21, strides=[1, 1, 1, 1], padding='SAME')
    L21 = Layer_Norm(L21)

    Lo = graph['output'] = tf.nn.tanh(L21) * 150 + 255./2

    return Lo, graph


def Layer_Norm(L):

    m, n_H, n_W, n_C = L.get_shape().as_list()

    # calculate mean and variance
    mu, sigma_sq = tf.nn.moments(L, [1,2], keep_dims=True)

    # define some variables
    shift = tf.Variable(tf.zeros([n_C]))
    scale = tf.Variable(tf.ones(n_C))
    epsilon = 1e-3

    # lets normalize the network layer
    normalized = (L - mu) / (sigma_sq + epsilon)**(.5)
    return scale * normalized + shift
