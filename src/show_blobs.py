# This function is used to display the blobs that are detected using the
# get_blob_loc function. It uses the OpenCV function cv.circle() to draw
# the circles on the image and the OpenCV function cv.imshow() to display
# the original image with the circles overlaid. The inputs to this function
# are:
#
# img - The main input image where blobs are detected.
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
# This function has no outputs.

# import OpenCV library

import cv2 as cv

# import NumPy library

import numpy as np

# import the get_blob_loc function

from get_blob_loc import get_blob_loc

def show_blobs(img,oct_num=3,DoG_layer_num=4,sigma=1.6,k=np.sqrt(2)):
    
    # compute the indices of the maxima of the difference of gaussian for
    # all octaves. Also, compute the layer number where these maxima appear
    # while also accounting for different octaves. This is done to
    # account for the different scales in each octave. For example, if a
    # maximum is detected in the second layer of the second octave, then
    # maxima_layer_num = the number of difference of gaussian layers in
    # each octave + 1.
    
    (maxima_idx,maxima_layer_num) = get_blob_loc(img,
                                    oct_num=oct_num,
                                    DoG_layer_num=DoG_layer_num,
                                    sigma=sigma,k=k)
    
    # compute vector to compute the different sigma values
    
    s = np.arange(DoG_layer_num)
    
    # compute the different k values based on consecutive powers
    
    k_vect = np.power(k,s)
    
    # compute an array of sigmas for each octave
    
    sigmas = np.tile(sigma*k_vect,(oct_num,1))
    
    # this variable is used to scale each row of the sigmas array
    # accordingly to account for down-sampling. For example, the first
    # layer in the second octave should have a sigma that is twice that of
    # the first layer in the first octave.
    
    sigmas_scaling = np.power(2,np.arange(oct_num).reshape((oct_num,1)))
    
    # perform scaling on each row
    
    sigmas = np.multiply(sigmas,sigmas_scaling)
    
    # compute the radii of the circles based on their characteristic
    # scales. A blob with a radius of sigma*sqrt(2) will be detected. The
    # array is also flattened to allow for advanced indexing later on.
        
    circle_radii = np.ceil(np.sqrt(2)*sigmas).astype(int).flatten()
    
    # the maxima_layer_num array is used as an index to match the
    # circle radii with the correct scale in scale space. For example, if
    # the number of desired octaves is 3, the number of DoG layers in
    # each octave is 4, then for a detected blob in the second layer in
    # the second octave will mean that maxima_layer_num = 1 + 4 = 5.
    # Hence, the 5th element of circle_radii will be chosen.
    
    circle_radii = list(circle_radii[maxima_layer_num])
    
    # overlay the red circles on the image
    
    for idx,radius in zip(maxima_idx,circle_radii):
    
        img2 = cv.circle(img,idx,radius,(0,0,255),2)
    
    # display the image
    
    cv.imshow('Final Image',img2)
