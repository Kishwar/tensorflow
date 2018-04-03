# These are my hobby project codes developed in python using OpenCV and TensorFlow
# Some of the projects are tested on Mac, Some on Raspberry Pi
# Anyone can use these codes without any permission
#
# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '01.04.18' '21:18'

# imports
from utils import *
from model import *
import time

def optimize(ContentImage, StyleImage, OutImage, content_weight, style_weight,
             tv_weight, vgg_path, epochs=4, print_iterations=4,
             learning_rate=1e-3):

    # Reset the graph
    tf.reset_default_graph()

    # Start interactive session
    sess = tf.InteractiveSession()

    # Create shape of (1, IW, IH, IC)
    img_shape = (1, IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS)

    # --------------------------------
    #   PRECOMPUTE CONTENT FEATURES  #
    # --------------------------------

    content_features = {}

    # lets have the tensor for Content Image(s)
    XContent = tf.placeholder(tf.float32, shape=img_shape, name="XContent")

    # lets normalize by subtracting mean
    TContentImages = normalize_image(XContent)

    ContImageModl = vgg19(vgg_path, TContentImages)

    content_features[CONTENT_LAYER] = ContImageModl[CONTENT_LAYER]

    # ---------------------------------
    #            NOISE IMAGE          #
    # ---------------------------------
    GenImage = generate_noise_image(XContent.get_shape())

    GenImageModl = vgg19(vgg_path, GenImage)

    content_size = tensor_size(content_features[CONTENT_LAYER])
    assert tensor_size(content_features[CONTENT_LAYER]) == tensor_size(GenImageModl[CONTENT_LAYER])

    # ---------------------------------------------------
    #  CONTENT LOSS FROM CONTENT IMAGE AND NOISE IMAGE  #
    # ---------------------------------------------------
    J_content = compute_content_cost(content_features[CONTENT_LAYER], GenImageModl[CONTENT_LAYER],
                                        content_weight, content_size)

    # ------------------------------
    #   PRECOMPUTE STYLE FEATURES  #
    # ------------------------------
    XStyle_shape = (1,) + StyleImage.shape
    print("Style shape = " + str(XStyle_shape))

    # lets have the tensor for Style Image
    XStyle = tf.placeholder(tf.float32, shape=XStyle_shape, name='style_image')

    # lets normalize by subtracting mean
    TStyleImage = normalize_image(XStyle)

    StyImageModl = vgg19(vgg_path, TStyleImage)

    NStyleImage = np.array([StyleImage])
    J_style = compute_style_cost(StyImageModl, GenImageModl, XStyle, NStyleImage, style_weight)

    J_tv = compute_tv_cost(GenImage, tv_weight, img_shape)

    # Total cost - we need to minimize this
    J = total_cost(J_content, J_style, J_tv)

    # define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # define train_step
    train_step = optimizer.minimize(J)

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    delta_time = 0

    for epoch in range(epochs):

        start_time = time.time()

        X_IContent = np.zeros(img_shape, dtype=np.float32)
        X_IContent[0] = getresizeImage(ContentImage[0]).astype(np.float32)

        assert X_IContent.shape == img_shape

        train_step.run(feed_dict={XContent:X_IContent})

        end_time = time.time()

        delta_time += (end_time - start_time)

        if ((epoch % print_iterations == 0) or ((epoch == epochs - 1) and (epochs > 2))):

            start_time = time.time()

            out = sess.run([J_style, J_content, J_tv, J, GenImage], feed_dict = {XContent:X_IContent})

            oJ_style, oJ_content, oJ_tv, oJ, oGenImage = out

            # save this iteration / epoch image
            save_image(OutImage + '/out_' + str(epoch) + '.jpg', oGenImage)

            end_time = time.time()

            delta_time += (end_time - start_time)

            print('Processing time %s for %s Epoch(s).' %(delta_time, print_iterations))

            print('Iteration: %d, J: %s, J_style: %s, J_content: %s, J_tv: %s' % (epoch, oJ, oJ_style, oJ_content, oJ_tv))

            delta_time = 0