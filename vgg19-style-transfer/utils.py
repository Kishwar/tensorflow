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
import cv2
import tensorflow as tf
from constants import *

def load_mat_file(file):
    return scipy.io.loadmat(file)

def resizeImage(image):
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

def gram_matrix(A):
    return tf.matmul(A, tf.transpose(A))

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to hav ethem of shape (n_C, n_H * n_W)
    a_S = tf.reshape(tf.transpose(a_S), [n_C, n_H*n_W])
    a_G = tf.reshape(tf.transpose(a_G), [n_C, n_H*n_W])

    # Compute gram matrics for both S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Compute the loss
    J_style_layer = (1 / (4 * (n_C**2) * (n_H * n_W)**2)) * (tf.reduce_sum(tf.square(tf.subtract(GS, GG))))

    return J_style_layer


def compute_style_cost(model, sess):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()        # get dimension of generated image (1, n_H, n_W, n_C)

    # Reshape a_C and a_G
    a_C_unrolled = tf.reshape(tf.transpose(a_C), [n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(tf.transpose(a_G), [n_H * n_W, n_C])

    # compute the cost
    J_content = (1 / (4 * n_H * n_W * n_C)) * (tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))))

    return J_content


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    J = (alpha * J_content) + (beta * J_style)

    return J

def reshape_and_normalize_image(image):
    """
    Reshape and normalize the input image (content or style)
    """

    # Reshape image to mach expected input
    image = np.reshape(image, ((1,) + image.shape))

    # Substract the mean to match the expected input
    image = image - MEANS

    return image

def generate_noise_image(content_image, noise_ratio = NOISE_RATIO):
    """
    Generates a noisy image by adding random noise to the content_image
    """

    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20, (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)

    return input_image

def save_image(path, image):

    # Un-normalize the image so that it looks good
    image = image + MEANS

    # Clip and Save the image
    image = np.clip(image[0], 0, 255).astype('uint8')

    scipy.misc.imsave(path, image)