# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '07.04.18' '07:49'

# imports
import scipy.io
import os
import tensorflow as tf
from constants import *
import functools

def load_mat_file(file):
    return scipy.io.loadmat(file)

def resizeImage(image):
    return scipy.misc.imresize(image, (IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS))

def getresizeImage(path):
    return resizeImage(scipy.misc.imread(path, mode='RGB'))

def list_files(path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break
    return files

def gram_matrix(A):
    # return np.matmul(np.transpose(A), A)
    return tf.matmul(tf.transpose(A), A)

def compute_layer_style_cost(StyImageModl, XStyle, StylImage):

    J_style_Layer = {}

    for layer, coeff in STYLE_LAYERS:

        # lets feed and eval XStyle tensor
        features = StyImageModl[layer].eval(feed_dict={XStyle:StylImage})

        # reshape features
        features = np.reshape(features, (-1, features.shape[3]))

        # compute gram matrix over it
        J_style_Layer[layer] = tf.divide(gram_matrix(features), features.size)

    return J_style_Layer

def compute_layer_gen_cost(GenImageModl):

    J_gen_Layer = {}

    for layer, coeff in STYLE_LAYERS:

        LGenImgModl = GenImageModl[layer]

        # Retrieve dimensions from a_G
        m, n_H, n_W, n_C = LGenImgModl.get_shape().as_list()

        features = tf.reshape(LGenImgModl, (m, n_H * n_W, n_C))

        features_T = tf.transpose(features, perm=[0,2,1])

        numerator = tf.matmul(features_T, features)
        denominator = (tf.multiply(tf.multiply(n_H, n_W), n_C))

        # print(numerator)
        # print(tf.cast(denominator, tf.float32))

        J_gen_Layer[layer] = tf.divide(tf.cast(numerator, tf.float32), tf.cast(denominator, tf.float32))

    return J_gen_Layer


def compute_style_cost(StyImageModl, GenImageModl, XStyle, StylImage, style_weight):

    # initialize the overall style cost
    J_style = []

    style_loss_layer = compute_layer_style_cost(StyImageModl, XStyle, StylImage)
    gen_loss_layer = compute_layer_gen_cost(GenImageModl)

    for layer, coeff in STYLE_LAYERS:

        loss_Style = style_loss_layer[layer]

        loss_Gen = gen_loss_layer[layer]

        J_style.append(tf.multiply(tf.to_float(2), tf.divide(tf.nn.l2_loss(tf.subtract(loss_Gen, loss_Style)), tf.cast(tf.reduce_prod(loss_Style.get_shape()), tf.float32))))

    J_style = tf.multiply(style_weight, functools.reduce(tf.add, J_style))

    return J_style

def compute_content_cost(a_C, a_G, CWt, CCz):
    """
    Computes the content cost
    """
    J_content = tf.multiply(CWt, tf.multiply(tf.to_float(2), tf.divide(tf.nn.l2_loss(tf.subtract(a_G, a_C)), CCz)))

    return J_content

def compute_tv_cost(GenImage, tv_weight, batch_shape):

    # total variation denoising
    tv_y_size = tensor_size(GenImage[:,1:,:,:])
    tv_x_size = tensor_size(GenImage[:,:,1:,:])
    y_tv = tf.nn.l2_loss(tf.subtract(GenImage[:,1:,:,:], GenImage[:,:batch_shape[1]-1,:,:]))
    x_tv = tf.nn.l2_loss(tf.subtract(GenImage[:,:,1:,:], GenImage[:,:,:batch_shape[2]-1,:]))

    J_tv = tf.multiply(tv_weight, tf.multiply(tf.to_float(2), tf.divide(tf.divide(x_tv, tf.add(tf.to_float(tv_x_size), y_tv)), tv_y_size)))

    return J_tv


def total_cost(J_content, J_style, J_tv):

    J = tf.add(tf.add(J_content, J_style), J_tv)

    return J

def normalize_image(image):
    """
    Normalize the input image (content or style)
    """

    # Substract the mean to match the expected input
    image = tf.subtract(image, MEANS)

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
    noise_image = np.random.uniform(-20, 20, (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS))

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

def get_video_image(image):

    # Un-normalize the image so that it looks good
    image = image + MEANS

    # Clip and Save the image
    image = np.clip(image[0], 0, 255).astype('uint8')

    return image