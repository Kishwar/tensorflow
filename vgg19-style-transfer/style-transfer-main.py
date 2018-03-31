# These are my hobby project codes developed in python using OpenCV and TensorFlow
# Some of the projects are tested on Mac, Some on Raspberry Pi
# Anyone can use these codes without any permission
#
# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '27.03.18' '17:52'

# imports
from model import *
import scipy.misc

def run():

    # Reset the graph
    tf.reset_default_graph()

    # Start interactive session
    sess = tf.InteractiveSession()

    print('loading model ...')
    model = load_vgg_model('./pre_trained_model/imagenet-vgg-verydeep-19.mat')
    print('model loaded ...')

    # print model as dicc elements
    print(model)

    ContentImage = scipy.misc.imread('./images/ContentImage.jpg')
    ContentImage = resizeImage(ContentImage)
    print('Content Image Shape = ' + str(ContentImage.shape))

    StyleImage = scipy.misc.imread('./images/StyleImage.jpg')
    StyleImage = resizeImage(StyleImage)
    print('Style Image Shape = ' + str(StyleImage.shape))

    ContentImage = reshape_and_normalize_image(ContentImage)
    print('Content Image Shape = ' + str(ContentImage.shape))

    StyleImage = reshape_and_normalize_image(StyleImage)
    print('Style Image Shape = ' + str(StyleImage.shape))

    generated_image = generate_noise_image(ContentImage)

    # Assign the content image to be the input of the VGG model.
    sess.run(model['input'].assign(ContentImage))

    # Select the output tensor of layer conv4_2
    out = model['conv4_2']

    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)


    # Assign the input of the model to be the "style" image
    sess.run(model['input'].assign(StyleImage))

    # Compute the style cost
    J_style = compute_style_cost(model, sess)

    # Compute total cost --> We need to optimise this cost
    J = total_cost(J_content, J_style, alpha = 10, beta = 40)

    # define optimizer
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

    # define train_step
    train_step = optimizer.minimize(J)

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Run the noisy input image (initial generated image) through the model.
    sess.run(model['input'].assign(generated_image))

    for i in range(NUM_OF_ITERATIONS):

        # Run the session on the train_step to minimize the total cost
        _ = sess.run(train_step)

        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Print every 20 iteration.
        if(i%20 == 0):

            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("./output/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image('output/generated_image.jpg', generated_image)


if __name__ == "__main__":
    run()