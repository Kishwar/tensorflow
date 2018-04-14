# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

__author__ = 'kishwarkumar'
__date__ = '07.04.18' '07:54'

from vgg19Model import *
from noiseModel import *
import time
import numpy as np
import cv2

def optimize(ContentImages, StyleImage, CheckPoint, chkpnt_epochs, content_weight, style_weight,
             tv_weight, vgg_path, epochs=4, print_iterations=4, learning_rate=1e1, batch_size=128):

    print('--------------------------------------------------------------------------------------------')
    print('content weight=%d, style weight=%d, tv weight=%d' %(content_weight, style_weight, tv_weight))
    print('epochs=%d, print iterations=%d, learning rate=%d' %(epochs, print_iterations, learning_rate))

    with tf.device("/gpu:0"):
    
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            # Create shape of (Batch Size, IH, IW, IC)
            XContentInputShape = (batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)
        
            # --------------------------------
            #   PRECOMPUTE CONTENT FEATURES  #
            # --------------------------------
        
            content_features = {}
        
            # lets have the tensor for Content Image(s)
            XContent = tf.placeholder(tf.float32, shape=XContentInputShape, name="XContent")
        
            # lets normalize by subtracting mean
            TContentImages = normalize_image(XContent)
        
            ContImageModl = vgg19(vgg_path, TContentImages)
        
            print('--------------------------------------------------------------------------------------------')
            print('optimize.py: Content VGG19 Layer [conv4_2] Shape: ' + str(ContImageModl[CONTENT_LAYER].get_shape()))
        
            content_features[CONTENT_LAYER] = ContImageModl[CONTENT_LAYER]
        
            # ---------------------------------
            #            NOISE IMAGE          #
            # ---------------------------------
            preds, Dpreds = noiseModel(XContent/255.0)
        
            print('--------------------------------------------------------------------------------------------')
            print('optimize.py: Noise Model Input Shape: ' + str(Dpreds['input'].get_shape()))
            print('optimize.py: Noise Model output Shape: '+ str(Dpreds['output'].get_shape()))
        
            GenImageModl = vgg19(vgg_path, normalize_image(preds))
        
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
        
            # lets have the tensor for Style Image
            XStyle = tf.placeholder(tf.float32, shape=XStyle_shape, name='style_image')
        
            # lets normalize by subtracting mean
            TStyleImage = normalize_image(XStyle)
        
            StyImageModl = vgg19(vgg_path, TStyleImage)
        
            NStyleImage = np.array([StyleImage])
            J_style = compute_style_cost(StyImageModl, GenImageModl, XStyle, NStyleImage, style_weight)
        
            J_tv = compute_tv_cost(preds, tv_weight, XContentInputShape)
        
            # Total cost - we need to minimize this
            J = total_cost(J_content, J_style, J_tv)
        
            # define optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)
        
            # define train_step
            train_step = optimizer.minimize(J)
        
            # Initialize global variables
            sess.run(tf.global_variables_initializer())
        
            delta_time = 0

            Updated = False
        
            print('--------------------------------------------------------------------------------------------')
            print('training started... %s files will be loaded.' %batch_size)
        
            for epoch in range(epochs):
        
                iterations = 0
                while iterations * batch_size < len(ContentImages):
        
                    start_time = time.time()

                    if ((len(ContentImages)  == batch_size) and (Updated ==False)):
                        X_IContent = np.zeros(XContentInputShape, dtype=np.float32)
        
                        curr = iterations * batch_size
                        step = curr + batch_size
        
                        for i, path in enumerate(ContentImages[curr:step]):
                            X_IContent[i] = getresizeImage(path)
        
                        assert X_IContent.shape == XContentInputShape

                        Updated = True

                        print('X_IContent Updated...')
        
                    # train_step.run(feed_dict={XContent:X_IContent})
                    sess.run(train_step, feed_dict={XContent:X_IContent})
        
                    end_time = time.time()
        
                    delta_time += (end_time - start_time)
        
                    iterations += 1
        
                    if len(ContentImages) < 500:  # why 500, I don't know..
                        printIt = ((epoch % print_iterations == 0) or (epoch % chkpnt_epochs == 0)) and (epoch > 0)
                    else:
                        printIt = (iterations % print_iterations == 0) or (epoch % chkpnt_epochs == 0) and (epoch > 0)
        
                    if printIt or ((epoch == epochs - 1) and (epochs > 2)):
        
                        start_time = time.time()
        
                        out = sess.run([J_content, J_style, J_tv, J, GenImageModl], feed_dict = {XContent:X_IContent})
        
                        oJ_content, oJ_style, oJ_tv, oJ, oGenImage = out
        
                        end_time = time.time()
        
                        delta_time += (end_time - start_time)
        
                        # print('Processing time %s for %s Epoch(s). %d%% finished.' %(delta_time, print_iterations,
                        #                                                             int(((iterations * batch_size)/len(ContentImages))*100)))
                        print('Processing time %s' %delta_time)
        
                        print('Iteration: %d, J: %s, J_style: %s, J_content: %s, J_tv: %s' % (epoch, oJ, oJ_style, oJ_content, oJ_tv))
        
                        delta_time = 0
        
                if epoch > 0 and epoch % chkpnt_epochs == 0:
                    print('Saving Noise Model...')
                    saver = tf.train.Saver()
                    saver.save(sess, CheckPoint + 'NoiseModel-' + str(epoch) + '-.ckpt')
                    print('Noise model saved..' + ' ' + 'NoiseModel-' + str(epoch) + '-.ckpt')


def generate(ContentImage, CheckPoint, Output, CamURL):

    # Check if we have cam or image as input
    if CamURL == '255':
        # TODO
        print('TODO')
    else:
        # get camera handle
        cap = cv2.VideoCapture(CamURL)
        
        while (True):
        
            g = tf.Graph()
            
            soft_config = tf.ConfigProto(allow_soft_placement=True)
            soft_config.gpu_options.allow_growth = True

            with g.as_default(), tf.device('/gpu:0'), tf.Session(config=soft_config) as sess:

                    # Initialize global variables
                    sess.run(tf.global_variables_initializer())

                    # Capture frame-by-frame
                    ret, frame = cap.read()
    
                    frame = resizeImage(frame)
    
                    # Create shape of (Batch Size, IH, IW, IC)
                    XContentInputShape = (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)
            
                    # lets have the tensor for Content Image(s)
                    XContent = tf.placeholder(tf.float32, shape=XContentInputShape, name="XContent")
            
                    preds, _ = noiseModel(XContent/255.0)
            
                    saver = tf.train.Saver()
            
                    X = np.zeros(XContentInputShape, dtype=np.float32)
                    X[0] = frame
            
                    if os.path.isdir(CheckPoint):
                        ckpt = tf.train.get_checkpoint_state(CheckPoint)
                        if ckpt and ckpt.model_checkpoint_path:
                            saver.restore(sess, ckpt.model_checkpoint_path)
                        else:
                            raise Exception("No checkpoint found...")
                    else:
                        saver.restore(sess, CheckPoint)
            
                    _pred = sess.run(preds, feed_dict={XContent: X})
            
                    # Display the resulting frame
                    # cv2.imshow('styled', get_video_image(_pred))
            
                    # Display the frame
                    # cv2.imshow('frame', frame)
            
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
