# This function computes the location of the maxima in a squared DoG
# octave. More precisely, it stacks every three layers in the octave
# and then computes a window around each pixel of the three layers
# simultaneously to compare the pixel in the middle layer to its 26
# neighbors in the middle layer and the adjacent layers. The input to this
# function is:
#
# ss_DoG_squared - An octave of DoG layers
#
# This function returns the indices of the maxima in the octave in
# maxima_idx and the DoG layer number where the maxima are located in
# maxima_layer_num. 

# import the NumPy library

import numpy as np

# import different functions from the scikit-image library to compute an
# appropriate global thresholding value 

from skimage import filters as flt

def get_maxima_loc(ss_DoG_squared):
    
    # initialize lists to store results
    
    maxima_idx = []
    
    maxima_layer_num = []
    
    for i in range(1,len(ss_DoG_squared)-1):
        
        # stack 3 consecutive layers together to prepare for non-max
        # suppression
        
        g = np.dstack(ss_DoG_squared[i-1:i+2])
        
        # compute a threshold value of the middle image for later
        # comparison. The method used here is Yen's method. However,
        # different thresholds obtained from different statistical methods
        # will yield different results for blob detection. For example,
        # the statistical mean can be used, such that the line below may
        # be substituted for the following line:
        #
        # thresh_layer = np.mean(g[:,:,1]).astype(int)
        #
        # The mean detects many of the blobs. However, it does
        # lead to many false positives. On the other hand, the line below
        # can also be substituted with the following line:
        #
        # thresh_layer = flt.threshold_otsu(g[:,:,1])
        #
        # To obtain a threshold value using Otsu's method. This method
        # is the second most precise out of the three methods. It
        # detects less blobs than the mean, but contains more false
        # positives than Yen's method.
        #
        # Yen's method is by far the most precise of the three methods.
        # However, it is sometimes susceptible to false negatives, where it
        # does not detect certain blobs.
        #
        # Note that Dr. Tianfu Wu was consulted before using these
        # built-in functions. Please refer to our conversation on Slack on
        # November 27th 2019 at 5:13 PM and 10:43 PM.
                
        thresh_layer = flt.threshold_yen(g[:,:,1])
        
        # pad the stacked images with 1 zero before and 1 zero after their
        # rows and columns to make room for a 3x3 sliding window
        
        padded_stack = np.zeros((g.shape[0]+2,g.shape[1]+2,g.shape[2]),
                                dtype=g.dtype)
        
        padded_stack[1:-1,1:-1,:] = g
        
        # iterate through the stack of layers
        
        for x in range(padded_stack.shape[0]-2):
            for y in range(padded_stack.shape[1]-2):
                
                # generate a 3x3x3 window around each pixel in the stack
                
                window = padded_stack[x:x+3,y:y+3,:]
                
                # compute the maximum value in the 3x3x3 window
                
                max_value = np.amax(window)
                
                # compare the value of the center pixel with the maximum
                # value to see if they are the same. Also, make sure that
                # the value of the center pixel is greater than the
                # threshold value to make sure it is a maximum.
                
                if (np.isclose(window[1,1,1],max_value) and
                   (window[1,1,1] > thresh_layer)):
                    
                    # append results to lists
                    
                    maxima_idx.append((x,y))
                    
                    maxima_layer_num.append(i)
        
    return (np.asarray(maxima_idx),np.asarray(maxima_layer_num))
