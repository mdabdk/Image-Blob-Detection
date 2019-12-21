# This function computes the squared Difference of Gaussian layers
# for a scale-space octave. The input to this function is:
#
# scale_space - a single octave in scale space

# import the NumPy library

import numpy as np

def get_DoG_squared(scale_space):
    
    # initialize list to store squared DoG layers
    
    DoG_squared = []
    
    # loop through each layer in the octave
    
    for i in range(len(scale_space)-1):
        
        # compute the difference of gaussians
        
        DoG_layer = np.subtract(scale_space[i+1],scale_space[i])
        
        # square the result
        
        DoG_layer_squared = np.power(DoG_layer,2)
        
        # append result to list
        
        DoG_squared.append(DoG_layer_squared)
    
    return DoG_squared
