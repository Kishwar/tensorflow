# These are my hobby project codes developed in python using OpenCV and TensorFlow
# Some of the projects are tested on Mac, Some on Raspberry Pi
# Anyone can use these codes without any permission
#
# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '25.03.18' '20:18'

#imports
import tensorflow as tf
import numpy as np
from datetime import datetime
from utils import *
from model import *

# globals
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "./data/", "Path to data files")
tf.flags.DEFINE_string("logs_dir", "./logs/", "Path to where log files are to be saved")
tf.flags.DEFINE_string("mode", "train", "mode: train (Default)/ test")

def main(argv=None):
    # data is downloaded from https://inclass.kaggle.com/c/facial-keypoints-detector/data
    # google, yahoo or facebook account is need to get data because sign-in required.

    # The train set consists of 3,761 grayscale images of 48x48 pixels in size and a 3,761
    # label set of seven elements each. Each element encodes an emotional stretch,
    # 0 = anger, 1 = disgust, 2 = fear, 3 = happy, 4 = sad, 5 = surprise, 6 = neutral.

    # lets read data
    train_images, train_labels, valid_images, valid_labels, test_images = \
                                        read_data(FLAGS.data_dir)
    print "Train size:      " + str(train_images.shape)
    print "Validation size: " + str(valid_images.shape)
    print "Test size:       " + str(test_images.shape)
    print "Train label size " + str(train_labels.shape)

    global_step = tf.Variable(0, trainable=False)
    dropout_prob = tf.placeholder(tf.float32)
    input_dataset = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1], name="input")
    input_labels = tf.placeholder(tf.float32, [None, NUM_LABELS])

    # get weight objects
    w1, w2, w3, w4, w_o = layer_weights()

    # get bias objects
    b1, b2, b3, b4, b_o = layer_biases()

    emo_model = model(input_dataset, w1, w2, w3, w4, w_o, b1, b2, b3, b4, b_o, p_keep_conv)

    # apply softmax on output layer of model
    o_emo_model = tf.nn.softmax(emo_model, name="output")

    loss_val = loss(emo_model, input_labels)
    train_op = train(loss_val, global_step)

    idxckp = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global idxckp
            saver.restore(sess, ckpt.model_checkpoint_path)
            idxckp = tf.train.latest_checkpoint(FLAGS.logs_dir)[-1]
            print "Model restored! ckpt:" + str(tf.train.latest_checkpoint('./logs/')[-1])

        for step in xrange(MAX_ITERATIONS):
            batch_image, batch_label = get_next_batch(train_images,\
                                                      train_labels, step)

            sess.run(train_op, feed_dict = {input_dataset: batch_image, \
                         input_labels: batch_label, p_keep_conv: 0.8})


            if step % 10 == 0:
                train_loss = sess.run(loss_val,\
                                feed_dict = {input_dataset: batch_image, \
                                input_labels: batch_label, p_keep_conv: 0.8})
                print "Training Loss: %f" % train_loss

            if step % 100 == 0:
                valid_loss = sess.run(loss_val, \
                                feed_dict = {input_dataset: valid_images,\
                                input_labels: valid_labels, p_keep_conv: 0.8})
                print "%s Validation Loss: %f" % (datetime.now(), valid_loss)

                saver.save(sess, FLAGS.logs_dir + 'model.ckpt', global_step=0)

if __name__ == "__main__":
    tf.app.run()