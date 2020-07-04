# This function computes the 2D FFT as an iteration of two 1D FFTs
#
# Tested 11/24/2019

from scipy import fftpack as dsp

def FFT_2D(img):
    
    # compute the 1D FFT of the input image row-wise
    
    fft_1D = dsp.fft(img,axis=0)
    
    # compute the 1D FFT of the already computed 1D FFT column-wise to
    # obtain the 2D FFT
    
    fft_2D = dsp.fft(fft_1D,axis=1)
    
    return fft_2D
