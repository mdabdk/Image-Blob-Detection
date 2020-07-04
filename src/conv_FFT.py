# This function convolves the inputs f and h using forward and inverse
# FFTs. The output g has the same shape as the the input f. For example,
# if f is an A x B array and h is a C x D array, then:
#
# g = conv_FFT(f,h,img_filter=False)
#
# will yield a an array g with a size of A x B. This function
# yields the same result as scipy.signal.convolve(f,h,mode='same'), where
# f and h are two 2D NumPy arrays.
# 
# Note that the img_filter flag is used to determine the output date type
# of this function if it will be used to filter an image. Hence, if
# img_filter = True, f is an input image, and h is the filter kernel,
# then the function will behave like the OpenCV function filter2D:
#
# cv.filter2D(img,-1,np.rot90(h,2),borderType=cv.BORDER_CONSTANT)
#
# Note that the filter kernel h was rotated by 180 degrees in this case
# because the cv.filter2D function actually computes correlation not
# convolution. For image filtering, the following use cases have been
# tested:
#
# - f.dtype = uint8, h.dtype = int32
# - f.dtype = uint8, h.dtype = float64
# - f.dtype = float64, h.dtype = float64
# - f.dtype = float64, h.dtype = int32
# 
# Tested 11/27/2019

# import NumPy library

import numpy as np

# import FFT_2D function

from FFT_2D import FFT_2D

# import iFFT_2D function

from iFFT_2D import iFFT_2D

# import the pad_img function

from pad_img import pad_img

# import the get_arr_ctr function

from get_arr_ctr import get_arr_ctr

def conv_FFT(f,h,img_filter=False):
        
    # minimum dimensions of padded arrays to avoid wrap-around error
    
    P = f.shape[0] + h.shape[0] - 1
    
    Q = f.shape[1] + h.shape[1] - 1
        
    # zero-pad arrays to ensure linear, and not circular, convolution in
    # spatial domain
    
    pad_size_f = (int(np.ceil((P-f.shape[0])/2)),
                  int(np.floor((P-f.shape[0])/2)),
                  int(np.ceil((Q-f.shape[1])/2)),
                  int(np.floor((Q-f.shape[1])/2)))
    
    pad_size_h = (int(np.ceil((P-h.shape[0])/2)),
                  int(np.floor((P-h.shape[0])/2)),
                  int(np.ceil((Q-h.shape[1])/2)),
                  int(np.floor((Q-h.shape[1])/2)))
    
    padded_f = pad_img(f,pad_size_f)
    
    padded_h = pad_img(h,pad_size_h)
    
    # compute the DFTs of the two zero-padded inputs
    
    F = FFT_2D(padded_f)
    
    H = FFT_2D(padded_h)
    
    # compute the element-wise product of the two DFTs, which is the same
    # as spatial convolution
    
    G = np.multiply(F,H)
    
    # compute the inverse FFT then take the real part and center it to
    # obtain the convolved result
    
    g = np.fft.ifftshift(np.real(iFFT_2D(G)))
            
    # check if f and h contain floating-point numbers to determine the
    # output data type
    
    f_isfloat = issubclass(f.dtype.type, np.float)
    
    h_isfloat = issubclass(h.dtype.type, np.float)
    
    # compute the indices of the centers of the f and g arrays
    
    f_ctr = get_arr_ctr(f.shape)
    
    g_ctr = get_arr_ctr(g.shape)
    
    # compute the top-left index used to crop out the convolved result
    
    out_start_idx = g_ctr - f_ctr
    
    # cropped output
    
    cropped_out = g[out_start_idx[0]:out_start_idx[0]+f.shape[0],
                    out_start_idx[1]:out_start_idx[1]+f.shape[1]]
    
    # the following if statements determine the data type of the output
    # based on the img_filter flag passed into the input argument and
    # based on the data types of f and h
    
    # if either f or h contain floating-point numbers and the function
    # isn't used to filter an image
    
    if (f_isfloat or h_isfloat) and img_filter == False:
        
        conv_out = cropped_out
    
    # if both f and h contain floating-point numbers and the function
    # is used to filter an image
    
    elif (f_isfloat and h_isfloat) and img_filter == True:
        
        conv_out = np.clip(cropped_out,0,255)
    
    # if f does not contain floating-point numbers and the function is
    # used to filter an image
    
    elif (not f_isfloat) and img_filter == True:
        
        conv_out = np.around(np.clip(cropped_out,0,255)).astype(np.uint8)
        
    # if neither f nor h contain floating-point numbers and if the
    # function isn't used to filter an image
    
    elif img_filter == False:
        
        conv_out = np.around(cropped_out).astype(f.dtype)
    
    # if all of the above fails
    
    else:
        
        conv_out = cropped_out
    
    return conv_out
