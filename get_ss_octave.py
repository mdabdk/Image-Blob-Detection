# This function returns a single octave in scale-space. The inputs to this
# function are:
#
# img - An MxN array, representing a grayscale image or a squared DoG array
# 
# n - The number of layers in the octave
#
# sigma_init - The first sigma value for the first layer in the octave
#
# k_init - the scaling factor for sigma
#
# This function returns the octave of scale-space images.

# import the NumPy library

import numpy as np

# import the get_gaussian_kernel() function

from get_gaussian_kernel import get_gaussian_kernel

# import the conv_FFT() function

from conv_FFT import conv_FFT

def get_ss_octave(img,n=5,sigma_init=1.6,k_init=np.sqrt(2)):
    
    # generate the k values for the octave
    
    k = np.power(k_init,np.arange(n))
    
    # generate the sigma values for the octave
    
    sigmas = list(sigma_init*k)
    
    # initialize list to store scale-space octave layers
    
    scale_space = []
    
    for sigma in sigmas: # loop through each layer
        
        # compute the new gaussian kernel
        
        gaussian_kernel = get_gaussian_kernel(sigma)
        
        # filter the image with the new gaussian kernel
        
        g_layer = conv_FFT(img,gaussian_kernel,img_filter=False)
        
        # append result to list
        
        scale_space.append(g_layer)
    
    return scale_space
