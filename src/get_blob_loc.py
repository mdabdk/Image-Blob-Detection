# This function computes the indices of the blobs in an image. The inputs
# to the function are:
#
# input_img - The main input image where blobs are detected.
#
# oct_num - The number of octaves desired in scale space. The default
# value was chosen based on the suggestions in section 11.7 of the
# 4th edition of the 'Digital Image Processing' book by Rafael C. Gonzalez.
#
# DoG_layer_num - The number of Difference of Gaussian layers desired
# in each octave. The default value was chosen based on the suggestions in
# section 11.7 of the 4th edition of the 'Digital Image Processing' book
# by Rafael C. Gonzalez.
#
# sigma - The initial sigma value to be used for blurring. The default
# value was chosen based on the suggestions in section 11.7 of the 4th
# edition of the 'Digital Image Processing' book by Rafael C. Gonzalez.
#
# k - The initial scaling factor for sigma. The default value was chosen
# based on the suggestions in section 11.7 of the 4th edition of the
# 'Digital Image Processing' book by Rafael C. Gonzalez.
#
# This function outputs maxima_idx, which is list of tuples representing
# the x and y co-ordinates of the center of the blobs. This function also
# outputs maxima_layer_num, which is a 1D array that contains the layer
# number in which each maximum is detected. For example, if the number
# of desired octaves is 3 and the number of DoG layers is 4, then a
# maximum detected in the second layer in the third octave will have
# maxima_layer_num = 1 + 8 = 9.

# import NumPy library

import numpy as np

# import OpenCV library

import cv2 as cv

# import the get_ss_octave() function

from get_ss_octave import get_ss_octave

# import the get_DoG_squared() function

from get_DoG_squared import get_DoG_squared

# import the get_maxima_loc() function

from get_maxima_loc import get_maxima_loc

def get_blob_loc(input_img,oct_num=3,DoG_layer_num=4,sigma=1.6,
                 k=np.sqrt(2)):
    
    # the number of layers in each octave in scale space
    
    n = DoG_layer_num + 1
    
    # convert the input image to grayscale
    
    img = cv.cvtColor(input_img,cv.COLOR_BGR2GRAY)
    
    # initialize lists to store results
    
    maxima_idx = []
    
    maxima_layer_num = []
    
    for i in range(oct_num): # loop through each octave
    
        # compute octave in scale space
        
        scale_space = get_ss_octave(img,n,sigma_init=sigma,k_init=k)
        
        # compute the squared difference of Gaussians from this octave
        
        DoG_squared = get_DoG_squared(scale_space)
        
        # compute the indices of the maxima of the squared difference of
        # Gaussians, and the scale space layers in which they appear
        
        (max_idx,max_layer_num) = get_maxima_loc(DoG_squared)
        
        # scale indices to account for down-sampling
        
        max_idx = max_idx*np.power(2,i)
        
        # shift layer number where maxima appear to account for different
        # octaves
        
        max_layer_num = max_layer_num + DoG_layer_num*i
        
        # store results in lists
        
        maxima_idx.append(max_idx)
        
        maxima_layer_num.append(max_layer_num)
        
        # down-sample the third image in scale-space and store it
        
        dsby2 = scale_space[2]
        
        img = dsby2[::2,::2]
    
    # convert the indices of the maxima to a list of (x,y) tuple
    # coordinates to use to draw circles on the original image
    
    maxima_idx = np.vstack(maxima_idx).tolist()
    
    maxima_idx = [tuple(i[::-1]) for i in maxima_idx]
    
    # concatenate the list of layer numbers to form a 1D array
    
    maxima_layer_num = np.hstack(maxima_layer_num)
    
    return (maxima_idx,maxima_layer_num)
