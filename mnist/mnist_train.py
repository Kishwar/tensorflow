# These are my hobby project codes developed in python using OpenCV and TensorFlow
# Some of the projects are tested on Mac, Some on Raspberry Pi
# Anyone can use these codes without any permission
#
# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '25.03.18' '10:18'

# imports
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# constants
batch_size     = 128
test_size      = 256
img_size       = 28
num_classes    = 10
MAX_ITERATIONS = 100

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


def run():
    # A placeholder variable, X, is defined for the input images.
    # The data type for this tensor is set to float32 and the shape
    # is set to [None, img_size, img_size, 1], where None means that
    # the tensor may hold an arbitrary number of images:
    X = tf.placeholder("float", [None, img_size, img_size, 1])

    # Then we set another placeholder variable, Y, for the true
    # labels associated with the images that were input data in
    # the placeholder variable X.
    Y = tf.placeholder("float", [None, num_classes])

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    trX = trX.reshape(-1, img_size, img_size, 1)
    teX = teX.reshape(-1, img_size, img_size, 1)
    print(trX.shape, teX.shape)                        # ((55000, 28, 28, 1), (10000, 28, 28, 1))

    # get weight objects
    w1, w2, w3, w4, w_o = layer_weights()

    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    # create model object
    py_x = model(X, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

    # define cost / loss
    Y_ = tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)
    cost = tf.reduce_mean(Y_)

    # define optimizer used for this task [learning rate=0.001,
    optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)

    # Finally, we define predict_op that is the index with the largest
    # value across dimensions from the output of the mode
    predict_op = tf.argmax(py_x, 1)

    idxckp = 0

    # lets start tensorflow session for training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./logs/')

        if ckpt and ckpt.model_checkpoint_path:
            global idxckp
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model Restored!"
            idxckp = tf.train.latest_checkpoint('./logs/')[-1]

        for i in range(MAX_ITERATIONS):   # 100 iterations
            training_batch = zip(range(0, len(trX), batch_size),
                              range(batch_size, len(trX)+1, batch_size))
            for start, end in training_batch:
                sess.run(optimizer, feed_dict={X: trX[start:end], Y: trY[start:end],
                                           p_keep_conv: 0.8, p_keep_hidden: 0.5})

            test_indices = np.arange(len(teX))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]

            print(i+int(idxckp)+1, np.mean(np.argmax(teY[test_indices], axis=1) == sess.run
                         (predict_op, feed_dict={X: teX[test_indices],
                                                 Y: teY[test_indices],
                                                 p_keep_conv: 1.0,
                                                 p_keep_hidden: 1.0})))

            saver.save(sess, './logs/' + 'model.ckpt', global_step=i+int(idxckp)+1)


if __name__ == "__main__":
    run()