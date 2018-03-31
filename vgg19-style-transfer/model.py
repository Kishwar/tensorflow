# These are my hobby project codes developed in python using OpenCV and TensorFlow
# Some of the projects are tested on Mac, Some on Raspberry Pi
# Anyone can use these codes without any permission
#
# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '31.03.18' '21:00'

# imports
from utils import *
from constants import *
import tensorflow as tf
import numpy as np

def load_vgg_model(path):
    """
    Returns a model for the purpose of 'painting' the picture.
    Takes only the convolution layer weights and wrap using the TensorFlow
    Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
    the paper indicates that using AveragePooling yields better results.
    The last few fully connected layers are not used.
    Here is the detailed configuration of the VGG model:
        0 is conv1_1 (3, 3, 3, 64)
        1 is relu
        2 is conv1_2 (3, 3, 64, 64)
        3 is relu
        4 is maxpool
        5 is conv2_1 (3, 3, 64, 128)
        6 is relu
        7 is conv2_2 (3, 3, 128, 128)
        8 is relu
        9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax
    """
    vgg = load_mat_file(path)

    vgg_layers = vgg['layers']

    def _weights(layer, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        """
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

    # Constructs the graph model.
    graph = {}

    X = graph['input'] = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)), dtype = 'float32')

    W, b = _weights(0, 'conv1_1')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L1 = tf.nn.conv2d(X, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L1 = graph['conv1_1']  = tf.nn.relu(L1)

    W, b = _weights(2, 'conv1_2')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L2 = tf.nn.conv2d(L1, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L2 = graph['conv1_2']  = tf.nn.relu(L2)
    L2 = graph['avgpool1'] = tf.nn.avg_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W, b = _weights(5, 'conv2_1')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L3 = tf.nn.conv2d(L2, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L3 = graph['conv2_1']  = tf.nn.relu(L3)

    W, b = _weights(7, 'conv2_2')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L4 = tf.nn.conv2d(L3, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L4 = graph['conv2_2']  = tf.nn.relu(L4)
    L4 = graph['avgpool2'] =  tf.nn.avg_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W, b = _weights(10, 'conv3_1')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L5 = tf.nn.conv2d(L4, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L5 = graph['conv3_1']  = tf.nn.relu(L5)

    W, b = _weights(12, 'conv3_2')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L6 = tf.nn.conv2d(L5, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L6 = graph['conv3_2']  = tf.nn.relu(L6)

    W, b = _weights(14, 'conv3_3')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L7 = tf.nn.conv2d(L6, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L7 = graph['conv3_2']  = tf.nn.relu(L7)

    W, b = _weights(16, 'conv3_4')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L8 = tf.nn.conv2d(L7, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L8 = graph['conv3_4']  = tf.nn.relu(L8)
    L8 = graph['avgpool3'] =  tf.nn.avg_pool(L8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W, b = _weights(19, 'conv4_1')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L9 = tf.nn.conv2d(L8, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L9 = graph['conv4_1']  = tf.nn.relu(L9)

    W, b = _weights(21, 'conv4_2')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L10 = tf.nn.conv2d(L9, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L10 = graph['conv4_2']  = tf.nn.relu(L10)

    W, b = _weights(23, 'conv4_3')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L11 = tf.nn.conv2d(L10, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L11 = graph['conv4_3']  = tf.nn.relu(L11)

    W, b = _weights(25, 'conv4_4')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L12 = tf.nn.conv2d(L11, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L12 = graph['conv4_4']  = tf.nn.relu(L12)
    L12 = graph['avgpool4'] =  tf.nn.avg_pool(L12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W, b = _weights(28, 'conv5_1')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L13 = tf.nn.conv2d(L12, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L13 = graph['conv5_1']  = tf.nn.relu(L13)

    W, b = _weights(30, 'conv5_2')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L14 = tf.nn.conv2d(L13, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L14 = graph['conv5_2']  = tf.nn.relu(L14)

    W, b = _weights(32, 'conv5_3')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L15 = tf.nn.conv2d(L14, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L15 = graph['conv5_3']  = tf.nn.relu(L15)

    W, b = _weights(34, 'conv5_4')
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    L16 = tf.nn.conv2d(L15, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    L16 = graph['conv5_4']  = tf.nn.relu(L16)
    L16 = graph['avgpool5'] =  tf.nn.avg_pool(L16, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # L17 = FC Layer - We don't need it here [FC-4096]
    # L18 = FC Layer - We don't need it here [FC-4096]
    # L19 = FC Layer - We don't need it here [FC-1000]
    # Output - Softmax

    # We are using Avg-pool instead of max-pool in this model
    return graph