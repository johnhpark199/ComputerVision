import os
import imageio
import numpy as np
import cv2 as cv
import rawpy
import math


def to_32bit_float(path):
    raw_image = np.load(path)
    dim_1 = len(raw_image)
    dim_2 = len(raw_image[0])
    i = 0
    # creating new array for 32 bit floating point image
    new_image = [[0] * dim_2 for i in range(dim_1)]
    # iterating through and modifying each value in the given array
    while i < dim_1:
        j = 0
        while j < dim_2:
            new_image[i][j] = (float(raw_image[i][j]) / float(((2 ** 16) - 1)))
            j += 1
        i += 1
    return new_image


def demosaic(path):
    """ Demosaics a raw 16-bit image captured using a Bayer pattern.
      Arguments:
      raw: the input raw data in 16-bit integer [HxW]
      Returns:
        The demosaiced image in 32-bit floating point [HxWx3]
    """
    # retreiving 32 bit floating point image
    im_32b = to_32bit_float(path)
    # demosaicing 32 bit floating point image
    demos_im = demosaic_helper(im_32b)
    return demos_im

# demosaics green parts of image
# always called before red/blue demosaicing
def demosaic_helper(image):

    dim_1 = len(image)
    dim_2 = len(image[0])

    # creating new 3 dimensional array
    image_arr = [[[0] * 3 for j in range(dim_2)] for i in range(dim_1)]
    # iterating through x dimension of image
    test = 0
    # iterating through y dimension of image (i = y)
    for i in range(0, dim_1):

        # iterating through x dimension of image (j = x)
        for j in range(0, dim_2):

            # determining if at a green pixel
            if ((i + j) % 2 == 0):
                # adding correct pixel value
                image_arr[i][j][1] = sum_vert_horiz_green(image, j, i)
            # else summing for red/blue pixels
            else:
                # if i % 2 = 0 must be processing on a red/green row
                if (i % 2 == 0):
                    # calculating horizontally calculated red pixels into correct position
                    image_arr[i][j][2] = sum_horiz(image, j, i)
                    # calculating vertically calculated blue pixels into correct position
                    image_arr[i][j][0] = sum_vert(image, j, i)
                # must be on blue/green row
                else:
                    # calculating vertically calculated red pixels into correct position
                    image_arr[i][j][2] = sum_vert(image, j, i)
                    # # calculating horizontally calculated blue pixels into correct position
                    image_arr[i][j][0] = sum_horiz(image, j, i)
    return image_arr


# summing vertical pixels to correct value (tested)
def sum_vert(image, x_loc, y_loc):
    bottom_edge = len(image) - 1
    # summing top row
    if (y_loc == 0):
        blue_val = image[y_loc + 1][x_loc]

    # summing bottom row
    elif (y_loc == bottom_edge):
        blue_val = image[y_loc - 1][x_loc]

    # if not top or bottom row
    else:
        blue_val = 0.5 * (image[y_loc - 1][x_loc] + image[y_loc + 1][x_loc])

    return blue_val


# summing horizontal values to correct pixel (tested)
def sum_horiz(image, x_loc, y_loc):
    right_edge = len(image[0]) - 1

    # if processing right edge
    if (x_loc == right_edge):
        red_val = image[y_loc][x_loc - 1]

    # any other case
    else:
        red_val = 0.5 * (image[y_loc][x_loc - 1] + image[y_loc][x_loc + 1])
    return red_val


# determines appropriate value to be returned for green aspect
# of 3-dimensional image array
def sum_vert_horiz_green(image, x_loc, y_loc):
    bottom_edge = len(image) - 1
    right_edge = len(image[0]) - 1
    # if processing a top left
    if (x_loc == 0 and y_loc == 0):
        green_val = 0.5 * (image[x_loc + 1][y_loc] + image[x_loc][y_loc + 1])

    # processing bottom right pixel
    elif (x_loc == right_edge and y_loc == bottom_edge):
        green_val = 0.5 * (image[y_loc][x_loc - 1] + image[y_loc - 1][x_loc])

    # bottom left corner (tested)
    elif (x_loc == 0 and y_loc == bottom_edge):
        green_val = 0.5 * (image[y_loc - 1][x_loc] + image[y_loc][x_loc + 1])

    # processing a top edge (tested)
    elif (y_loc == 0):
        green_val = 0.3333 * (image[y_loc][x_loc - 1] + image[y_loc][x_loc + 1] + image[y_loc + 1][x_loc])

    # processing a left edge sum (tested)
    elif (x_loc == 0):

        green_val = 0.3333 * (image[y_loc][x_loc + 1] + image[y_loc - 1][x_loc] + image[y_loc + 1][x_loc])

    # pixel on right edge (tested)
    elif (x_loc == right_edge):

        green_val = 0.3333 * (image[y_loc - 1][x_loc] + image[y_loc + 1][x_loc] + image[y_loc][x_loc - 1])

    # summing on bottom edge (tested)
    elif (y_loc == bottom_edge):
        green_val = 0.3333 * (image[y_loc - 1][x_loc] + image[y_loc][x_loc + 1] + image[y_loc][x_loc - 1])

    # sum is not on an edge (tested)
    else:
        green_val = 0.25 * (image[y_loc][x_loc - 1] + image[y_loc - 1][x_loc] + image[y_loc + 1][x_loc] + image[y_loc][x_loc + 1])
    return green_val


def white_balance(image):
    """ White balanaces a 32-bit floating point demosaiced image.
      This is done by simply scaling each channel so that its mean = 0.5.
      Arguments:
        image: the input image in 32-bit floating point [HxWx3]
      Returns:
        The white balanced image in 32-bit floating point [HxWx3]
    """
    # computing initial mean of each column
    mean_array = np.mean(image, axis = 0)
    # computing mean of columns of columns, resulting in array of size 3 with mean of colors
    total_mean = np.mean(mean_array, axis = 0)
    # finding value to divide by to give a mean of 0.5
    scaling_factors = total_mean / (.5)
    # apply balancing by dividing by scaling factor
    white_balanced_image = image / scaling_factors
    return white_balanced_image


def curve_and_quantize(image, inv_gamma=0.85):
    """ Applies inverse gamma function and quantizes to 8-bit.
      Arguments:
        image: the input image in 32-bit floating point [HxWx3]
        inv_gamma: the value of 1/gamma
      Returns:
        The curved and quantized image in 8-bit unsigned integer [HxWx3]
    """
    # curving image using inverse gamma filter
    curved_im = [[[(element ** inv_gamma) for element in sublist] for sublist in subsublist] for subsublist in image]
    # clipping image
    clipped_im = np.clip(curved_im, 0, 1)
    # casting list to array to make sure image is processed correctly
    float_array = np.array(clipped_im)
    # casting array to 8 bit int unsigned integer
    im_8bit = (float_array * 255).astype(np.uint8)
    return im_8bit
