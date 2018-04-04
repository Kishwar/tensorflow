# These are my hobby project codes developed in python using OpenCV and TensorFlow
# Some of the projects are tested on Mac, Some on Raspberry Pi
# Anyone can use these codes without any permission
#
# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '27.03.18' '22:56'

# imports
import scipy.io
import os
import tensorflow as tf
from constants import *
import functools

def load_mat_file(file):
    return scipy.io.loadmat(file)

def resizeImage(image):
    if not (len(image.shape) == 3 and image.shape[2] == 3):
        image = np.dstack((image,image,image))
    return np.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')

def getresizeImage(path):
    return resizeImage(scipy.misc.imread(path))

def list_files(path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break
    return files

def gram_matrix(A):
    # return tf.matmul(tf.transpose(A), A)
    return np.matmul(np.transpose(A), A)

def compute_layer_style_cost(StyImageModl, XStyle, StylImage):

    J_style_Layer = {}

    for layer, coeff in STYLE_LAYERS:

        # lets feed and eval XStyle tensor
        features = StyImageModl[layer].eval(feed_dict={XStyle:StylImage})

        # reshape features
        features = np.reshape(features, (-1, features.shape[3]))

        # compute gram matrix over it
        J_style_Layer[layer] = coeff * (gram_matrix(features)  / features.size)

    return J_style_Layer

def compute_layer_gen_cost(GenImageModl):

    J_gen_Layer = {}

    for layer, coeff in STYLE_LAYERS:

        LGenImgModl = GenImageModl[layer]

        # Retrieve dimensions from a_G
        m, n_H, n_W, n_C = LGenImgModl.get_shape().as_list()

        features = tf.reshape(LGenImgModl, (m, n_H * n_W, n_C))

        features_T = tf.transpose(features, perm=[0,2,1])

        J_gen_Layer[layer] = tf.matmul(features_T, features) / (n_H * n_W * n_C)

    return J_gen_Layer


def compute_style_cost(StyImageModl, GenImageModl, XStyle, StylImage, style_weight):

    # initialize the overall style cost
    J_style = []

    style_loss_layer = compute_layer_style_cost(StyImageModl, XStyle, StylImage)
    gen_loss_layer = compute_layer_gen_cost(GenImageModl)

    for layer, coeff in STYLE_LAYERS:

        loss_Style = style_loss_layer[layer]

        loss_Gen = gen_loss_layer[layer]

        J_style.append(2 * tf.nn.l2_loss(loss_Gen - loss_Style) / loss_Style.size)

    J_style = style_weight * functools.reduce(tf.add, J_style)

    return J_style

def compute_content_cost(a_C, a_G, CWt, CCz):
    """
    Computes the content cost
    """

    J_content = CWt * (2 * tf.nn.l2_loss(a_G - a_C) / CCz)

    return J_content

def compute_tv_cost(GenImage, tv_weight, batch_shape):

    # total variation denoising
    tv_y_size = tensor_size(GenImage[:,1:,:,:])
    tv_x_size = tensor_size(GenImage[:,:,1:,:])
    y_tv = tf.nn.l2_loss(GenImage[:,1:,:,:] - GenImage[:,:batch_shape[1]-1,:,:])
    x_tv = tf.nn.l2_loss(GenImage[:,:,1:,:] - GenImage[:,:,:batch_shape[2]-1,:])

    J_tv = tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size)

    return J_tv


def total_cost(J_content, J_style, J_tv):

    J = J_content + J_style + J_tv

    return J

def normalize_image(image):
    """
    Normalize the input image (content or style)
    """

    # Substract the mean to match the expected input
    image = image - MEANS

    return image

def Denormalize(image):
    """
    Denormalize the input image (content or style)
    """

    # Substract the mean to match the expected input
    image = image + MEANS

    return image

def generate_noise_image(content_image, noise_ratio = NOISE_RATIO):
    """
    Generates a noisy image by adding random noise to the content_image
    """

    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20, (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)

    return tf.Variable(tf.convert_to_tensor(input_image, dtype=tf.float32))

def save_image(path, image):

    # Un-normalize the image so that it looks good
    image = image + MEANS

    # Clip and Save the image
    image = np.clip(image[0], 0, 255).astype('uint8')

    scipy.misc.imsave(path, image)

# temp
def tensor_size(tensor):
    from operator import mul
    import functools
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)