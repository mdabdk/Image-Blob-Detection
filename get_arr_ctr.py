# Given a a 2-element tuple representing the dimensions of a 2D array as
# an input, this function returns the indices of the center of the array
# as a 2-element NumPy array, where the first element represents the row
# number and the second element represents the column number. For example:
#
# get_arr_ctr((3,3))
#
# would return (1,1), which means that the center of this 2D array is
# located at row number 1 and column number 1

# import NumPy library

import numpy as np

def get_arr_ctr(arr_shape):
    
    # if both dimensions are odd
    
    if ((arr_shape[0] % 2) and
        (arr_shape[1] % 2)): 
    
        arr_ctr = np.array([int(arr_shape[0]//2),
                            int(arr_shape[1]//2)])
    
    # if the number of rows is odd and the number of columns is even
    
    elif ((arr_shape[0] % 2) and not
          (arr_shape[1] % 2)):
        
        arr_ctr = np.array([int(arr_shape[0]//2),
                            int(arr_shape[1]//2)-1])
    
    # if the number of rows is even and the number of columns is odd
    
    elif (not (arr_shape[0] % 2) and
              (arr_shape[1] % 2)):
        
        arr_ctr = np.array([int(arr_shape[0]//2)-1,
                            int(arr_shape[1]//2)])
    
    # if both dimensions are even
    
    elif (not (arr_shape[0] % 2) and not
              (arr_shape[1] % 2)):
        
        arr_ctr = np.array([int(arr_shape[0]//2)-1,
                            int(arr_shape[1]//2)-1])
    
    return arr_ctr
