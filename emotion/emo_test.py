# These are my hobby project codes developed in python using OpenCV and TensorFlow
# Some of the projects are tested on Mac, Some on Raspberry Pi
# Anyone can use these codes without any permission
#
# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '26.03.18' '21:21'

# imports
import tensorflow as tf
import numpy as np
import cv2
from constants import *
from model import *

# globals
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "./data/", "Path to data files")
tf.flags.DEFINE_string("logs_dir", "./logs/", "Path to where log files are to be saved")
tf.flags.DEFINE_string("mode", "train", "mode: train (Default)/ test")

emotion = {0:'anger', 1:'disgust',\
           2:'fear',3:'happy',\
           4:'sad',5:'surprise',6:'neutral'}

def main(argv=None):

    input_dataset = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1], name="input")

    global_step = tf.Variable(0, trainable=False)
    dropout_prob = tf.placeholder(tf.float32)
    input_dataset = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1], name="input")
    input_labels = tf.placeholder(tf.float32, [None, NUM_LABELS])

    # get weight objects
    w1, w2, w3, w4, w_o = layer_weights()

    # get bias objects
    b1, b2, b3, b4, b_o = layer_biases()

    # load model structure
    emo_model = model(input_dataset, w1, w2, w3, w4, w_o, b1, b2, b3, b4, b_o, p_keep_conv)

    # lets start tensorflow session for training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./logs/')

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model loaded! ckpt:" + str(tf.train.latest_checkpoint('./logs/'))

             # load image
            image = cv2.imread('./HappyFace.jpg')

            # convert to gray scale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # resize
            input_img = np.resize(gray_image, (48, 48, 1))

            # let's predict
            predict_op = tf.argmax(emo_model, 1)
            print(emotion[predict_op.eval(feed_dict={input_dataset: [input_img], p_keep_conv: 0.8}, session=sess)[0]])

        else:
            exit()

if __name__ == "__main__":
    tf.app.run()