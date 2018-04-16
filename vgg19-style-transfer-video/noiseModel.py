# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

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

    # Convolution block 4
    W4 = layer_weights(3, 128, L3.get_shape().as_list()[3])
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 2, 2, 1], padding='SAME')
    L4 = Layer_Norm(L4)
    L4 = graph['conv3']  = tf.nn.relu(L4)

    # -------------------------------------------------------------
    #                         RESIDUAL BLOCKS                     #
    # -------------------------------------------------------------
    # Residual block 1
    W5 = layer_weights(3, 128, L4.get_shape().as_list()[3])
    L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
    L5 = Layer_Norm(L5)
    L5 = tf.nn.relu(L5)

    W6 = layer_weights(3, 128, L5.get_shape().as_list()[3])
    L6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME')
    L6 = Layer_Norm(L6)

    L7 = graph['res1'] = L4 + L6
    L7 = tf.nn.relu(L7)

    # Residual block 2
    W8 = layer_weights(3, 128, L7.get_shape().as_list()[3])
    L8 = tf.nn.conv2d(L7, W8, strides=[1, 1, 1, 1], padding='SAME')
    L8 = Layer_Norm(L8)
    L8 = tf.nn.relu(L8)

    W9 = layer_weights(3, 128, L8.get_shape().as_list()[3])
    L9 = tf.nn.conv2d(L8, W9, strides=[1, 1, 1, 1], padding='SAME')
    L9 = Layer_Norm(L9)

    L10 = graph['res2'] = L7 + L9
    L10 = tf.nn.relu(L10)

    # Residual block 3
    W11 = layer_weights(3, 128, L10.get_shape().as_list()[3])
    L11 = tf.nn.conv2d(L10, W11, strides=[1, 1, 1, 1], padding='SAME')
    L11 = Layer_Norm(L11)
    L11 = tf.nn.relu(L11)

    W12 = layer_weights(3, 128, L11.get_shape().as_list()[3])
    L12 = tf.nn.conv2d(L11, W12, strides=[1, 1, 1, 1], padding='SAME')
    L12 = Layer_Norm(L12)

    L13 = graph['res3'] = L10 + L12
    L13 = tf.nn.relu(L13)

    # Residual block 4
    W14 = layer_weights(3, 128, L13.get_shape().as_list()[3])
    L14 = tf.nn.conv2d(L13, W14, strides=[1, 1, 1, 1], padding='SAME')
    L14 = Layer_Norm(L14)
    L14 = tf.nn.relu(L14)

    W15 = layer_weights(3, 128, L14.get_shape().as_list()[3])
    L15 = tf.nn.conv2d(L14, W15, strides=[1, 1, 1, 1], padding='SAME')
    L15 = Layer_Norm(L15)

    L16 = graph['res4'] = L13 + L15
    L16 = tf.nn.relu(L16)

    # Residual block 5
    W17 = layer_weights(3, 128, L16.get_shape().as_list()[3])
    L17 = tf.nn.conv2d(L16, W17, strides=[1, 1, 1, 1], padding='SAME')
    L17 = Layer_Norm(L17)
    L17 = tf.nn.relu(L17)

    W18 = layer_weights(3, 128, L17.get_shape().as_list()[3])
    L18 = tf.nn.conv2d(L17, W18, strides=[1, 1, 1, 1], padding='SAME')
    L18 = Layer_Norm(L18)

    L19 = graph['res5'] = L16 + L18
    L19 = tf.nn.relu(L19)

    # -------------------------------------------------------------
    #                   CONV TRANSPOSE BLOCKS                     #
    # -------------------------------------------------------------
    # ConT block 1
    W20 = layer_weights(3, 64, L19.get_shape().as_list()[3], True)
    m, n_H, n_W, n_C = L19.get_shape().as_list()
    m, n_H, n_W, n_C = m, int(n_H * 2), int(n_W * 2), n_C

    L20 = tf.nn.conv2d_transpose(L19, W20, tf.stack([m, n_H, n_W, 64]),
                                 strides=[1, 2, 2, 1], padding='SAME')
    L20 = Layer_Norm(L20)
    L20 = tf.nn.relu(L20)

    # ConT block 2
    W21 = layer_weights(3, 64, L20.get_shape().as_list()[3], True)
    m, n_H, n_W, n_C = L20.get_shape().as_list()
    m, n_H, n_W, n_C = m, int(n_H * 2), int(n_W * 2), n_C

    L21 = tf.nn.conv2d_transpose(L20, W21, tf.stack([m, n_H, n_W, 64]),
                                 strides=[1, 2, 2, 1], padding='SAME')
    L21 = Layer_Norm(L21)
    L21 = tf.nn.relu(L21)

    # ConT block 3
    W22 = layer_weights(3, 32, L21.get_shape().as_list()[3], True)
    m, n_H, n_W, n_C = L21.get_shape().as_list()
    m, n_H, n_W, n_C = m, int(n_H * 2), int(n_W * 2), n_C
    L22 = tf.nn.conv2d_transpose(L21, W22, tf.stack([m, n_H, n_W, 32]),
                                 strides=[1, 2, 2, 1], padding='SAME')
    L22 = Layer_Norm(L22)
    L22 = tf.nn.relu(L22)

    # -------------------------------------------------------------
    #                       CONVOLUTION BLOCKS                    #
    # -------------------------------------------------------------
    # Convolution block 1
    W23 = layer_weights(9, 3, L22.get_shape().as_list()[3])
    L23 = tf.nn.conv2d(L22, W23, strides=[1, 1, 1, 1], padding='SAME')
    L23 = Layer_Norm(L23)

    Lo = graph['output'] = tf.nn.tanh(L23) * 150 + 255./2

    return Lo, graph


def Layer_Norm(L):

    # calculate mean and variance
    mu, sigma_sq = tf.nn.moments(L, [1,2], keep_dims=True)

    # define epsilon constant
    epsilon = 1e-3

    # lets normalize the network layer
    normalized = (L - mu) / (sigma_sq + epsilon)**(.5)
    return normalized
