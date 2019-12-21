# This file is used to run the main program. The following parameters can
# be adjusted for different results.

# import OpenCV library

import cv2 as cv

# import NumPy library

import numpy as np

# import the show_blobs function

from show_blobs import show_blobs

# Number of desired octaves in scale space

number_of_octaves = 3

# Number of desired layers of difference of Gaussian in each octave

number_of_DoG_layers = 4

# initial sigma value used for scale space

sigma = 1.6

# initial scaling factor for sigma

k = np.sqrt(2)

# read the input image. Please enter different image file names here to
# test different images

img = cv.imread('butterfly.jpg')

# display the detected blobs

show_blobs(img,number_of_octaves,number_of_DoG_layers,sigma,k)