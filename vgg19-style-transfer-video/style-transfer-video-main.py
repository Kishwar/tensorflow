# These are my hobby project codes developed in python using OpenCV and TensorFlow
# Some of the projects are tested on Mac, Some on Raspberry Pi
# Anyone can use these codes without any permission
#
# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#

# License
# Copyright (c) 2016 Logan Engstrom. Contact me for commercial use (or rather any use that is not academic research)
# (email: engstrom at my university's domain dot edu). Free for research use, as long as proper attribution is given
# and this copyright notice is retained.

__author__ = 'kishwarkumar'
__date__ = '06.04.18' '11:50'

# imports
from argparse import ArgumentParser
from optimize import *

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--style', type=str,
                        dest='style', help='path to the style image',
                        metavar='STYLE_IMAGE', required=True)

    parser.add_argument('--content', type=str,
                        dest='train_path', help='path to the content image',
                        metavar='CONTENT / TRAIN_PATH', required=True)

    parser.add_argument('--out', type=str,
                        dest='out_path', help='path to the output directory',
                        metavar='OUTDIR', required=True)

    parser.add_argument('--chkpnt', type=str,
                        dest='checkpoint', help='path to store noise model',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--print-iterations', type=int,
                        dest='print_iterations', help='print on terminal after these number of iterations',
                        metavar='CHECKPOINT_ITERATIONS', default=PRINT_ITERATIONS)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',
                        help='batch size (default %(default)s)',
                        metavar='BATCH SIZE', default=BATCH_SIZE)
    return parser

if __name__ == "__main__":
    parser = build_parser()
    Args = parser.parse_args()

    # lets get the style image
    StyleImage = getresizeImage(Args.style)    # np array
    #print('\n')
    #print('--------------------------------------------------------------------------------------------')
    #print('main: Style Image   - ' + str(Args.style))

    # lets get content images, stack all the paths together
    LContentImages = list_files(Args.train_path)
    ContentImages = [os.path.join(Args.train_path, CI) for CI in LContentImages]

    # args
    args = [
        ContentImages,
        StyleImage,
        Args.out_path,
        Args.checkpoint,
        Args.content_weight,
        Args.style_weight,
        Args.tv_weight,
        Args.vgg_path
    ]

    # kwargs
    kwargs = {
        "epochs": Args.epochs,
        "print_iterations": Args.print_iterations,
        "learning_rate": Args.learning_rate,
        "batch_size": Args.batch_size
    }

    optimize(*args, **kwargs)