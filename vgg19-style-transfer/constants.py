# These are my hobby project codes developed in python using OpenCV and TensorFlow
# Some of the projects are tested on Mac, Some on Raspberry Pi
# Anyone can use these codes without any permission
#
# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '27.03.18' '23:05'

# imports
import numpy as np

# constants
IMAGE_WIDTH = 256 # 400
IMAGE_HEIGHT = 256 #300
COLOR_CHANNELS = 3
MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
NOISE_RATIO = 0.256
LEARNING_RATE = 1e-3
NUM_OF_ITERATIONS = 200000

STYLE_LAYERS = [
    ('conv1_1', 1),
    ('conv2_1', 1),
    ('conv3_1', 1),
    ('conv4_1', 1),
    ('conv5_1', 1)]
CONTENT_LAYER = 'conv4_2'

TRAIN_PATH = 'data/train2014'
VGG_PATH = 'pre_trained_model/imagenet-vgg-verydeep-19.mat'
NUM_EPOCHS = 2
CHECKPOINT_ITERATIONS = 2
BATCH_SIZE = 4

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

