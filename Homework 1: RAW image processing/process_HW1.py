import rawpy
from rawpy import *
import argparse
import imageio
import os
from raw import *


parser = argparse.ArgumentParser()
parser.add_argument('path',help='path to NPY file')
args = parser.parse_args()

# this is the path to the output JPEG
path_out = os.path.basename(args.path).split('.')[0]+'.JPG'

image = demosaic(args.path)
# white balancing image
balance_im = white_balance(image)
# curving and quantizing image
processed_im = curve_and_quantize(balance_im)
imageio.imwrite(path_out, processed_im)