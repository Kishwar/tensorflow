# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '18.04.18' '21:54'

# imports
import tensorflow as tf
from tensorflow.contrib import rnn
from constants import *

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# MODEL
def RNN(x, weights, biases):

    # The current input data will be (batch_size, n_steps, n_input)
    # The required shape is a n_steps tensors list of shape (batch_size, n_input)
    x = tf.transpose(x, [1, 0, 2])          # 128,28,28 -> 28,28,128

    x = tf.reshape(x, [-1, n_input])        # 28 * 128, 28

    x = tf.split(axis=0,
                 num_or_size_splits=n_steps,
                 value=x)                   # 128, 28

    # Define a single LSTM cell: The BasicLSTMCell method defines LSTM recurrent
    # network cell. The forget_bias parameter is set to 1.0 to reduce the scale of
    # forgetting in the beginning of the training:
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)    # n_hidden = 128

    # Build the network: The rnn() operation creates the compute nodes
    # for a given amount of time steps:
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # The resulting tensor of the RNN function is a vector of length 10 for
    # determining which of the 10 classes the input image belongs to:
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

