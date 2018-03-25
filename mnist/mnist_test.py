# These are my hobby project codes developed in python using OpenCV and TensorFlow
# Some of the projects are tested on Mac, Some on Raspberry Pi
# Anyone can use these codes without any permission
#
# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '25.03.18' '13:13'

# imports
import tensorflow as tf
import numpy as np
import cv2
from model import *

def run():
    # A placeholder variable, X, is defined for the input images.
    # The data type for this tensor is set to float32 and the shape
    # is set to [None, img_size, img_size, 1], where None means that
    # the tensor may hold an arbitrary number of images:
    X = tf.placeholder("float", [None, img_size, img_size, 1], name="input")

    # get weight objects
    w1, w2, w3, w4, w_o = layer_weights()

    # create same model object for testing
    py_x = model(X, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

    # load image
    image = cv2.imread('./numbers/1.jpg')

    # convert to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # resize
    input_img = np.resize(gray_image, (img_size, img_size, 1))

    # lets start tensorflow session for training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./logs/')

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model loaded! ckpt:" + str(tf.train.latest_checkpoint('./logs/')[-1])

        predict_op = tf.argmax(py_x, 1)
        print(predict_op.eval(feed_dict={X: [input_img], p_keep_conv: 0.8, p_keep_hidden: 0.5}, session=sess)[0])


if __name__ == "__main__":
    run()