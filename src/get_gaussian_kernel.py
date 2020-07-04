# This function accepts a standard deviation as an inputs and outputs a
# spatial-domain normalized Gaussian window to be used for filtering.
# For example:
#
# get_gaussian_kernel(2)
#
# returns a square kernel with odd dimensions consisting of a Gaussian
# pulse at the center with a standard deviation of 2.
#
# Tested 11/25/2019

# import NumPy library

import numpy as np

# import NDimage library

from scipy import ndimage

def get_gaussian_kernel(sigma):
    
    # kernel width & height
    
    m = np.ceil(6*sigma).astype(int)
    
    # make sure m is odd
    
    if not (m % 2):
        
        m = m + 1
        
    # initialize impulse response shape
    
    kernel = np.zeros((m,m))
    
    # compute the indices of the center of the window
    
    kernel_center = (np.floor(m/2).astype(int),
                     np.floor(m/2).astype(int))
    
    # set impulse at center
    
    kernel[kernel_center] = 1
    
    # compute impulse response
    
    kernel = ndimage.gaussian_filter(kernel,sigma)
    
    return kernel
