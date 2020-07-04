# This function computes the 2D inverse FFT by first finding the 2D FFT of
# the complex conjugate of the frequency spectrum, then conjugating that
# and dividing it by the area of the original image (M x N)
#
# Tested 11/24/2019

from FFT_2D import FFT_2D

def iFFT_2D(spec):
    
    # obtain conjugate frequency spectrum
    
    conj_img = FFT_2D(spec.conjugate())
    
    # obtain output image
    
    img_out = (conj_img.conjugate())/(spec.shape[0]*spec.shape[1])
    
    return img_out
