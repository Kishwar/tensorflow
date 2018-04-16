# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '07.04.18' '07:50'

# imports
import numpy as np

# constants
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
COLOR_CHANNELS = 3
MEANS = np.array([123.68, 116.779, 103.939])   #.reshape((1,1,1,3))
NOISE_RATIO = 0.256
LEARNING_RATE = 1e-3
NUM_OF_ITERATIONS = 1
WEIGHTS_INIT_STDEV = .1

STYLE_LAYERS = [
    ('conv1_1', 1),
    ('conv2_1', 1),
    ('conv3_1', 1),
    ('conv4_1', 1),
    ('conv5_1', 1)]
CONTENT_LAYER = 'conv4_2'

VGG_PATH = 'pre_trained_model/imagenet-vgg-verydeep-19.mat'
NUM_EPOCHS = 4
PRINT_ITERATIONS = 4
CHECKPOINT_ITERATIONS = 1
BATCH_SIZE = 16

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2
