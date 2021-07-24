import numpy as np
import pywt
from utils.helpers import equidistance
import matplotlib.pyplot as plt


def forward_fft(x, y, n=64, hamming=True):
    """Fuction to convert coordinates to fft coefs
    n: number of fft coefficients to keep for x and y
    Returns fft_x and fft_y with len of n+1 (DC component and n fft coefs)
    """
    assert len(x) == len(y)

    start = len(x) // 3
    x = np.concatenate(np.repeat([x], 10, axis=0))  # [start : start + 10 * len(x)]
    y = np.concatenate(np.repeat([y], 10, axis=0))
    """
    # repeating xx times
    start = len(x) // 3  # np.random.randint(len(x)) #
    x = np.concatenate((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x))[start : start + 20 * len(x)]
    y = np.concatenate((y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y))#[start : start + 10 * len(y)]
    """
    if hamming:
        x = x * np.hamming(len(x))
        y = y * np.hamming(len(y))
    fft_x = np.fft.fft(x)
    fft_y = np.fft.fft(y)

    assert np.allclose(np.conjugate(fft_x[-n:][::-1]), fft_x[1 : 1 + n])
    assert np.allclose(np.conjugate(fft_y[-n:][::-1]), fft_y[1 : 1 + n])

    return fft_x[:n], fft_y[:n]


def inverse_fft(fft_x, fft_y, hamming=True):
    """Fuction to convert fft coefficients back to coordinates
    n: number of data points
    """
    n = len(fft_x)
    assert len(fft_x) == len(fft_y)
    if n is None:
        fft_x = np.concatenate((fft_x, np.conjugate(fft_x[1:][::-1])))
        fft_y = np.concatenate((fft_y, np.conjugate(fft_y[1:][::-1])))
    else:
        n = len(fft_x)
        fft_x = np.concatenate(
            (
                fft_x,
                np.zeros((10 * 2 * n - 2 * n + 1), dtype="complex128"),
                # np.zeros((18 * n + 1), dtype="complex128"),
                np.conjugate(fft_x[1:][::-1]),
            )
        )
        fft_y = np.concatenate(
            (
                fft_y,
                np.zeros((10 * 2 * n - 2 * n + 1), dtype="complex128"),
                # np.zeros((18 * n + 1), dtype="complex128"),
                np.conjugate(fft_y[1:][::-1]),
            )
        )
    ix = np.fft.ifft(fft_x)
    iy = np.fft.ifft(fft_y)

    if hamming:
        ix = ix / np.hamming(len(ix))
        iy = iy / np.hamming(len(iy))

    if n is not None:
        """
        start = n // 3
        ix = ix[2 * n - start : 2 * n - start + 100]  # Take the middle part of ix
        iy = iy[2 * n : 2 * n + 100] #iy[2 * n + 2 * n - start : 2 * n + 4 * n - start]  # iy[2 * n :2 * n + 2*n]  # Take the middle part of iy
        """
        ix = ix[4 * n : 4 * n + 2 * n]
        iy = iy[4 * n : 4 * n + 2 * n]
    """
    return (
        np.concatenate((ix, ix))[len(ix) - len(ix) // 3 : 2 * len(ix) - len(ix) // 3],
        iy,
    )  # np.append(ix, ix[0]), np.append(iy, iy[0])
    """
    return ix, iy


def fourier_coeffs(shape_coords, n=64):
    coords = shape_coords

    x = np.array([p[0] for p in coords])
    y = np.array([p[1] for p in coords])
    # aligning start point of contour
    start = np.argmax(y)
    x = np.concatenate((x,x))[start : start+ len(coords)]
    y = np.concatenate((y,y))[start : start+ len(coords)]
    
    x_, y_ = equidistance(x, y, n_points=2 * n)

    fft_x, fft_y = forward_fft(x_, y_, n=n)  # returns len(fft_x)=len(fft_y)=n

    coeffs = [fft_x] + [fft_y]

    ix_, iy_ = inverse_fft(fft_x, fft_y)
    ix, iy = equidistance(ix_.real, iy_.real, len(coords))
    """
    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    ax[0].plot(x, y)
    ax[0].axis("scaled")
    ax[1].plot(x, label="x coord")
    ax[1].plot(y, label="y coord")
    ax[1].scatter(x[0], y[0])
    ax[1].legend()
    # ax[2].plot(np.concatenate((x_,x_,x_))[len(x_)//3 :  len(x_)//3 + 2*len(y_)], label = "x coord")
    # ax[2].plot(np.concatenate((y_,y_)), label = "y coord")
    ax[3].plot(ix, iy)
    ax[3].scatter(ix[0], iy[0])
    ax[3].axis("scaled")
    plt.tight_layout()
    """
    error = (np.average(abs(x - ix)) + np.average(abs(y - iy))) / 2

    return coeffs, error


def lowpassfilter(signal, thresh=0.7, wavelet="db4"):
    thresh = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per")
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signal


def forward_wavelet(x, y, wavelet_type="db5"):
    cAx, cDx = pywt.dwt(x, wavelet_type)
    cAy, cDy = pywt.dwt(y, wavelet_type)
    # cAx = pywt.threshold(cAx, np.std(cAx)/2, mode='soft')
    # cAy = pywt.threshold(cAy, np.std(cAy)/2, mode='soft')
    return cAx, cAy


def inverse_wavelet(cAx, cAy, wavelet_type="db5"):
    ix_ = pywt.idwt(cAx, None, wavelet_type, "smooth")
    iy_ = pywt.idwt(cAy, None, wavelet_type, "smooth")
    return ix_, iy_


def wavelet_coefs(shape, n=64):
    coords = shape

    start = np.random.randint(len(coords))

    x = np.array([p[0] for p in coords])
    # x = np.concatenate(np.repeat([x], 2, axis=0))[start : start + len(x)]
    y = np.array([p[1] for p in coords])
    # y = np.concatenate(np.repeat([y], 2, axis=0))[start : start + len(y)]

    x_, y_ = equidistance(x, y, n_points=2 * n)

    cAx, cAy = forward_wavelet(x_, y_)

    coeffs = [cAx] + [cAy]

    ix_, iy_ = inverse_wavelet(cAx, cAy)

    ix, iy = equidistance(ix_, iy_, len(coords))

    """
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    #ax[0].imshow(shape)
    ax[0].plot(x,y)
    ax[0].scatter(x[0], y[0], color='r')
    ax[0].axis('scaled')
    ax[1].plot(x, label = "x coord")
    ax[1].plot(y, label = "y coord")
    ax[1].legend()
    ax[2].plot(ix,iy)
    ax[2].scatter(ix[0], iy[0], color='r')
    ax[2].axis('scaled')
    plt.tight_layout()
    """
    error = (np.average(abs(x - ix)) + np.average(abs(y - iy))) / 2
    return coeffs, error
