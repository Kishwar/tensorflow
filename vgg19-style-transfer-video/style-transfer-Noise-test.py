# Contact info: Kishwar Kumar [kumar.kishwar@gmail.com]
# Country: Germany
#


__author__ = 'kishwarkumar'
__date__ = '11.04.18' '21:52'

# imports
from argparse import ArgumentParser
from optimize import *

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--chkpnt', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--in-path', type=str,
                        dest='in_path',help='file to transform',
                        metavar='IN_PATH', required=True)

    parser.add_argument('--out-path', type=str,
                        dest='out_path', help='output directory path', metavar='OUT_PATH',
                        required=True)

    return parser

if __name__ == "__main__":
    parser = build_parser()
    Args = parser.parse_args()

    # lets get the input image
    ContentImage = getresizeImage(Args.in_path)    # np array

    # args
    args = [
        ContentImage,
        Args.checkpoint_dir,
        Args.out_path
            ]

    # kwargs
    kwargs = {}

    generate(*args, **kwargs)