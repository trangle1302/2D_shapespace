import numpy as np
import pywt
from utils.helpers import equidistance, find_nearest, find_centroid
import matplotlib.pyplot as plt
import pyefd

def forward_fft(x, y, n=64, hamming=False, repeat=False):
    """Fuction to convert coordinates to fft coefs
    n: number of fft coefficients to keep for x and y
    Returns fft_x and fft_y with len of n+1 (DC component and n fft coefs)
    """
    assert len(x) == len(y)

    if repeat:
        # start = len(x) // 3
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


def inverse_fft(fft_x, fft_y, hamming=False, repeat=False):
    """Fuction to convert fft coefficients back to coordinates
    n: number of data points
    """
    assert len(fft_x) == len(fft_y)
    if not repeat:
        fft_x = np.concatenate((fft_x, [0], np.conjugate(fft_x[1:][::-1])))
        fft_y = np.concatenate((fft_y, [0], np.conjugate(fft_y[1:][::-1])))
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

    if repeat:
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


### separate fft of cell and nucleus, and add back offset to 1st DC of nucleus
def fourier_coeffs(shape_coords, n=64):
    coords = shape_coords

    x = np.array([p[0] for p in coords])
    y = np.array([p[1] for p in coords])

    # aligning start point of contour
    centroid = find_centroid(coords)
    _, val = find_nearest(y[np.where(x > centroid[0])], centroid[1])
    if len(np.where(y == val)[0]) > 1:
        largest_x = x.min()
        current_idx = None
        for idx in np.where(y == val)[0]:
            if x[idx] > largest_x:
                largest_x = x[idx]
                current_idx = idx
        idx = current_idx
    else:
        idx = np.where(y == val)[0][0]

    x = np.concatenate((x, x))[idx : idx + len(coords)]
    y = np.concatenate((y, y))[idx : idx + len(coords)]

    x_, y_ = equidistance(x, y, n_points=2 * n)

    fft_x, fft_y = forward_fft(x_, y_, n=n)  # returns len(fft_x)=len(fft_y)=n

    coeffs = [fft_x] + [fft_y]

    ix_, iy_ = inverse_fft(fft_x, fft_y)
    ix, iy = equidistance(ix_.real, iy_.real, len(coords))
    """
    fig, ax = plt.subplots(1, 3, figsize=(6, 3))
    ax[0].plot(x, y)
    ax[0].axis("scaled")
    ax[1].plot(x, label="x coord")
    ax[1].plot(y, label="y coord")
    ax[1].legend()
    ax[2].plot(ix, iy)
    ax[2].scatter(ix[0], iy[0], c="r")
    ax[2].axis("scaled")
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

def forward_efd(xy, n=64):
    """
    Calculate elliptic fourier descriptors for a closed contour (2D)

    Parameters
    ----------
    xy : list of tuples
        coordinates/array of all points in contour, N x 2.
    n : int, optional
        order of fourier coeffs to calculate. The default is 64.

    Returns
    -------
    coeffs : ndarray of n x 4
        n x 4 of [a_n, b_n, c_n, d_n].
    a0, c0 : float
        A_0, C_0 coefficients.

    """
    # aligning start point of contour
    centroid = find_centroid(xy)
    x = np.array([p[0] for p in xy])
    y = np.array([p[1] for p in xy])
    _, val = find_nearest(y[np.where(x > centroid[0])], centroid[1])
    if len(np.where(y == val)[0]) > 1:
        largest_x = x.min()
        current_idx = None
        for idx in np.where(y == val)[0]:
            if x[idx] > largest_x:
                largest_x = x[idx]
                current_idx = idx
        idx = current_idx
    else:
        idx = np.where(y == val)[0][0]

    xy = np.concatenate((xy, xy))[idx : idx + len(xy)]
    coeffs = pyefd.elliptic_fourier_descriptors(xy, order=n)
    a0, c0 = pyefd.calculate_dc_coefficients(xy)
    return coeffs, a0, c0

def backward_efd(a0_c0_coeffs, n_points=64):
    """
    Reconstruct shape based on elliptic fourier descriptors

    Parameters
    ----------
    coeffs : ndarray of n_terms x 4
        elliptical fourier descriptors array.     
    a0, c0 : float
        elliptic locus in [#a]_ and [#b]_.
    n_points : int
        number of points for reconstructed contour.

    Returns
    -------
    xy_t : [n_points, 2]
        A list of x,y coordinates for the reconstructed contour.

    """
    n_terms = len(a0_c0_coeffs[2:])//4
    #print(len(a0_c0_coeffs), n_terms)
    coeffs = np.array(a0_c0_coeffs[2:]).reshape((n_terms,4))
    a0 = a0_c0_coeffs[0]
    c0 = a0_c0_coeffs[1]
    xy_t = pyefd.reconstruct_contour(coeffs, locus=(a0,c0), num_points = n_points)
    return xy_t[:,0], xy_t[:,1]

def elliptical_fourier_coeffs(shape_coords, n=64, plot=False):
    coords = shape_coords
    # elliptical fourier descriptors are rotation invariant, so no need to align start point of contour

    e_coefs, a0, c0 = forward_efd(coords, n=n)  # returns [n x 4], a0, c0
    coeffs = [a0] + [c0] + e_coefs.ravel().tolist()
    #print(coeffs[:4])
    i_coords = backward_efd(coeffs, n_points=len(shape_coords))
    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(6, 3))
        ax[0].plot(coords[:,0], coords[:,1], c= "b", label="original")
        ax[0].axis("scaled")
        ax[2].scatter(coords[0,0], coords[0,1], c="r")
        ax[1].legend()
        ax[2].plot(i_coords[:,0], i_coords[:,1], c= "orange", label="reconstructed")
        ax[2].scatter(i_coords[0,0], i_coords[0,1], c="r")
        ax[2].axis("scaled")
        plt.tight_layout()

    error = np.mean(abs(coords - i_coords))

    return coeffs, error