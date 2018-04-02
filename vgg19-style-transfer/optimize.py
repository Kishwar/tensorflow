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

def optimize(ContentImages, StyleImage, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=2,
             batch_size=4, save_path='saver/fns.ckpt', learning_rate=1e-3):

    # Reset the graph
    tf.reset_default_graph()

    # Start interactive session
    sess = tf.InteractiveSession()

    mod = len(ContentImages) % batch_size
    if mod > 0:
        print('Train set has been trimmed slightly..')
        ContentImages = ContentImages[:-mod]
        print('New total number of Content Images = ' + str(len(ContentImages)))

    # update batch shape
    batch_shape = (batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS)
    print('batch shape =' + str(batch_shape))

    # --------------------------------
    #   PRECOMPUTE CONTENT FEATURES  #
    # --------------------------------

    content_features = {}

    # lets have the tensor for Content Image(s)
    XContent = tf.placeholder(tf.float32, shape=batch_shape, name="XContent")

    # lets normalize by subtracting mean
    TContentImages = normalize_image(XContent)

    ContImageModl = vgg19(vgg_path, TContentImages)

    content_features[CONTENT_LAYER] = ContImageModl[CONTENT_LAYER]

    # ---------------------------------
    #            NOISE IMAGE          #
    # ---------------------------------
    GenImage = generate_noise_image(XContent.get_shape())

    GenImageModl = vgg19(vgg_path, GenImage)

    content_size = tensor_size(content_features[CONTENT_LAYER]) * batch_size
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

    J_tv = compute_tv_cost(GenImage, tv_weight, batch_shape)

    # Total cost - we need to minimize this
    J = total_cost(J_content, J_style, J_tv)

    # define optimizer
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

    # define train_step
    train_step = optimizer.minimize(J)

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        num_examples = len(ContentImages)
        iterations = 0
        while iterations * batch_size < num_examples:
            start_time = time.time()

            curr = iterations * batch_size
            step = curr + batch_size
            X_batch = np.zeros(batch_shape, dtype=np.float32)

            for j, bImg in enumerate(ContentImages[curr:step]):
                X_batch[j] = getresizeImage(bImg).astype(np.float32)

            iterations += 1
            assert X_batch.shape[0] == batch_size

            train_step.run(feed_dict={XContent:X_batch})

            end_time = time.time()

            delta_time = end_time - start_time

            # print("delta_time: %s, iteration: %s, print_iterations %s" % (delta_time, iterations, print_iterations))

            is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples

            if ((iterations % print_iterations == 0) or is_last):

                out = sess.run([J_style, J_content, J_tv, J, GenImage], feed_dict = {XContent:X_batch})

                oJ_style, oJ_content, oJ_tv, oJ, oGenImage = out

                # save model
                saver = tf.train.Saver()
                res = saver.save(sess, save_path)

                # lets put losses/costs together
                J_All = (oJ_style, oJ_content, oJ_tv, oJ)

                oGenImage = Denormalize(oGenImage)

                print('Epoch %d, Iteration: %d, J: %s, J_style: %s, J_content: %s, J_tv: %s' % (epoch, iterations, oJ, oJ_style, oJ_content, oJ_tv))

                # yield(oGenImage, J_All, iterations, epoch)