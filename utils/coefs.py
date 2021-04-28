import numpy as np
import pywt
from utils.helpers import equidistance
import matplotlib.pyplot as plt

def forward_fft(x, y, n=64):
    """ Fuction to convert coordinates to fft coefs
    n: number of fft coefficients to keep for x and y
    Returns fft_x and fft_y with len of n+1 (DC component and n fft coefs)
    """
    assert len(x) == len(y)
    fft_x = np.fft.fft(x)
    fft_y = np.fft.fft(y)
    
    assert np.allclose(np.conjugate(fft_x[-n:][::-1]), fft_x[1:1+n])
    assert np.allclose(np.conjugate(fft_y[-n:][::-1]), fft_y[1:1+n])
    
    return fft_x[:n+1], fft_y[:n+1]

def inverse_fft(fft_x, fft_y, n=None):
    """ Fuction to convert fft coefficients back to coordinates
    n: number of data points  
    """
    assert len(fft_x) == len(fft_y)
    if n is None:
        fft_x = np.concatenate((fft_x, np.conjugate(fft_x[1:][::-1])))
        fft_y = np.concatenate((fft_y, np.conjugate(fft_y[1:][::-1])))
    else:
        fft_x = np.concatenate((fft_x, np.zeros((2*n)), np.conjugate(fft_x[1:][::-1])))
        fft_y = np.concatenate((fft_y, np.zeros((2*n)), np.conjugate(fft_y[1:][::-1])))
    ix = np.fft.ifft(fft_x)
    iy = np.fft.ifft(fft_y)
    
    return ix, iy

def fourier_coeffs(shape_coords, n=64):
    coords = shape_coords

    x = np.array([p[0] for p in coords])
    y = np.array([p[1] for p in coords])

    x_, y_ = equidistance(x, y, n_points=n*2+1)
    fft_x, fft_y = forward_fft(x_,y_, n=n) #returns len(fft_x)=len(fft_y)=n+1
    
    coeffs = [fft_x] + [fft_y]
    
    ix_, iy_ = inverse_fft(fft_x, fft_y)
    ix, iy = equidistance(ix_.real, iy_.real, len(coords))
    '''
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].plot(x,y)
    ax[0].axis('scaled')
    ax[1].plot(x, label = "x coord")
    ax[1].plot(y, label = "y coord")
    ax[1].legend()
    ax[2].plot(ix.real,iy.real)
    ax[2].axis('scaled')
    plt.tight_layout()
    '''
    error = (np.average(abs(x - ix)) + np.average(abs(y - iy))) / 2

    return coeffs, error


def lowpassfilter(signal, thresh=0.7, wavelet="db4"):
    thresh = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per")
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signal


def wavelet_coefs(shape, wavelet_type="db5"):
    coords = find_contours(shape)

    x = np.array([p[0] for p in coords[0]])
    y = np.array([p[1] for p in coords[0]])

    cAx, cDx = pywt.dwt(x, wavelet_type)
    cAy, cDy = pywt.dwt(y, wavelet_type)

    # cAx = pywt.threshold(cAx, np.std(cAx)/2, mode='soft')
    # cAy = pywt.threshold(cAy, np.std(cAy)/2, mode='soft')

    ix = pywt.idwt(None, cDx, wavelet_type)
    iy = pywt.idwt(None, cDy, wavelet_type)

    coeffs = []
    """
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    #ax[0].imshow(shape)
    ax[0].plot(x,y)
    ax[0].axis('scaled')
    ax[1].plot(x, label = "x coord")
    ax[1].plot(y, label = "y coord")
    ax[1].legend()
    ax[2].plot(ix,iy)
    ax[2].axis('scaled')
    plt.tight_layout()
    """
    error = (np.average(abs(x - ix)) + np.average(abs(y - iy))) / 2
    return coeffs, error.real
