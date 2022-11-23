import numpy as np
from scipy import interpolate as spinterp
import warnings
import matplotlib.pyplot as plt
import scipy
from typing import Optional, List, Dict, Tuple
from pathlib import Path

def parameterize_image_coordinates(
    seg_mem: np.array, seg_nuc: np.array, lmax: int, nisos: List
):

    """
    Runs the parameterization for a cell represented by its spherical
    harmonics coefficients calculated by using ther package aics-shparam.
    Parameters
    --------------------
    seg_mem: np.array
        2D binary cell segmentation.
    seg_nuc: np.array
        2D binary nuclear segmentation.
    lmax: int
        Degree of fft expansion.
    nisos : list
        [a,b] representing the number of layers that will be used to
        parameterize the nucleoplasm and cytoplasm.
    Returns
    -------
    coords: np.array
        Array of shape 3xNxM, where NxM is the size of a
        parameterized intensity representation generated with
        same parameters lmax and nisos.
    coeffs_mem: dict
        coefficients that represent cell shape.
    centroid_mem: tuple
        (x,y) representing cell centroid.
    coeffs_nuc: dict
        coefficients that represent nuclear shape.
    centroid_nuc: tuple
        (x,y) representing nuclear centroid.
    """

    if (seg_mem.dtype != np.uint8) or (seg_nuc.dtype != np.uint8):
        warnings.warn(
            "One or more input images is not an 8-bit image\
        and will be cast to 8-bit."
        )

    # Cell SHE coefficients
    (coeffs_mem, _), (_, _, _, centroid_mem) = shparam.get_shcoeffs(
        image=seg_mem, lmax=lmax, sigma=0, compute_lcc=True, alignment_2d=False
    )

    # Nuclear SHE coefficients
    (coeffs_nuc, _), (_, _, _, centroid_nuc) = shparam.get_shcoeffs(
        image=seg_nuc, lmax=lmax, sigma=0, compute_lcc=True, alignment_2d=False
    )

    # Get Coordinates
    coords = get_mapping_coordinates(
        coeffs_mem=coeffs_mem,
        centroid_mem=centroid_mem,
        coeffs_nuc=coeffs_nuc,
        centroid_nuc=centroid_nuc,
        nisos=nisos,
    )

    # Shift coordinates to the center of the input
    # segmentations
    coords += np.array(centroid_mem).reshape(3, 1, 1)

    return coords, (coeffs_mem, centroid_mem, coeffs_nuc, centroid_nuc)


def get_interpolators(
    coeffs_mem: Dict,
    centroid_mem: Dict,
    coeffs_nuc: Dict,
    centroid_nuc: Dict,
    nisos: List,
):

    """
    Creates 1D interpolators for fft/wavelet coefficients with fixed points
    at: 1) nuclear centroid, 2) nuclear shell and 3) cell membrane.
    Also creates an interpolator for corresponding centroids.
    Parameters
    --------------------
    coeffs_mem: dict
        coefficients that represent cell shape (mem=membrane).
    centroid_mem: tuple
        (x,y) representing cell centroid
    coeffs_nuc: dict
        coefficients that represent nuclear shape (nuc=nuclear).
    centroid_nuc: tuple
        (x,y) representing nuclear centroid
    nisos : list
        [a,b] representing the number of layers that will be used to
        parameterize the nucleoplasm and cytoplasm.
    Returns
    -------
        coeffs_interpolator: spinterp.interp1d
        centroids_interpolator: spinterp.interp1d
        lmax: int
    """
    if len(coeffs_mem) != len(coeffs_nuc):
        raise ValueError(
            f"Number of cell and nuclear coefficients\
        do not match: {len(coeffs_mem)} and {len(coeffs_nuc)}."
        )

    # Total number of coefficients
    nc = len(coeffs_mem)
    # Degree of the expansion (lmax)
    lmax = int(np.sqrt(nc / 2.0) - 1)

    # Concatenate centroid into same array for interpolation
    centroids = np.c_[centroid_nuc, centroid_nuc, centroid_mem]

    # Array to be interpolated
    coeffs_ctr_arr = np.array([0 if i else 1 for i in range(nc)])
    coeffs_mem_arr = np.zeros((2, lmax + 1, lmax + 1))
    coeffs_nuc_arr = np.zeros((2, lmax + 1, lmax + 1))
    # Populate cell and nuclear arrays and concatenate into a single arr
    for k, kname in enumerate(["C", "S"]):
        for L in range(lmax + 1):
            for m in range(lmax + 1):
                coeffs_mem_arr[k, L, m] = coeffs_mem[f"fftcoeffs_L{L}M{m}{kname}"]
                coeffs_nuc_arr[k, L, m] = coeffs_nuc[f"fftcoeffs_L{L}M{m}{kname}"]
    coeffs_mem_arr = coeffs_mem_arr.flatten()
    coeffs_nuc_arr = coeffs_nuc_arr.flatten()
    coeffs = np.c_[coeffs_ctr_arr, coeffs_nuc_arr, coeffs_mem_arr]

    # Calculate fixed points for interpolation
    iso_values = [0.0] + nisos
    iso_values = np.cumsum(iso_values)
    iso_values = iso_values / iso_values[-1]

    # Coeffs interpolator
    coeffs_interpolator = spinterp.interp1d(iso_values, coeffs)

    # Centroid interpolator
    centroids_interpolator = spinterp.interp1d(iso_values, centroids)

    return coeffs_interpolator, centroids_interpolator, lmax


def make_kernel(center, k=3):
    """
    Generate kernel of size kxk around center in cartesian coord

    Parameters
    ----------
    center : int
        center point of the kernel.
    k : odd integer
        kernel size. The default is 3.

    Returns
    -------
    kernel : np.array of size (k,k)
    
    """
    assert k % 2 !=0 #assert k is odd
    step = k//2
    kernel = []
    c = [i for i in range(center-step, center+step+1,1)]
    for s in range(-step, step+1,1):
        kernel += [np.array(c) + s]
    kernel = np.array(kernel).reshape((k,k))
    return kernel

def kernel_coordinates(center_coord, k=3):
    assert k % 2 !=0 #assert k is odd
    step = k//2
    x,y = center_coord[0],center_coord[1]
    kernel_coords = []    
    for x_ in range(x-step, x+step+1,1):
        for y_ in range(y-step, y+step+1,1):      
            kernel_coords += [(x_,y_)]
    #kernel_coords = np.array(kernel_coords).reshape((k,k))
    return kernel_coords

    

def get_coordinates(nuc, mem, centroid, n_isos = [3,7], plot=True):
    """
    Creates 1D interpolators for x, y with fixed points
    at: 1) nuclear centroid, 2) nuclear shell and 3) cell membrane.
    
    Parameters
    --------------------
    nuc: list
        coefficients that represent nuclear shape (nuc=nuclear).
    mem: list
        coefficients that represent cell shape (mem=membrane).
    centroid: tuple
        (x,y) representing nucleus centroid (center to interpolate outward)
    n_isos : list
        [a,b] representing the number of layers that will be used to
        parameterize the nucleoplasm and cytoplasm.
    Returns
    -------
        ix: interpolated x values
        iy: interpolated y values
    """
    x_n, y_n = nuc[:len(nuc)//2], nuc[len(nuc)//2:]
    x_c, y_c = mem[:len(mem)//2], mem[len(mem)//2:]
    iso_values = [0.0] + n_isos
    iso_values = np.cumsum(iso_values)
    iso_values = iso_values / iso_values[-1]
    
    x = np.c_[np.full_like(x_n, centroid[0]), x_n, x_c]
    y = np.c_[np.full_like(y_n, centroid[1]), y_n, y_c]
    
    # Create x and y interpolator
    x_interpolator = scipy.interpolate.interp1d(iso_values, x)
    y_interpolator = scipy.interpolate.interp1d(iso_values, y)
    ix_list = []
    iy_list = []
    if plot:
        plt.plot(x_n, y_n)
        plt.plot(x_c, y_c)
        for i, iso_value in enumerate(np.linspace(0.0, 1.0, 1 + np.sum(n_isos))):
            # Get coeffs at given fixed point
            ix = x_interpolator(iso_value)
            iy = y_interpolator(iso_value)
            plt.plot(ix,iy, "--")
            plt.text(ix[i], iy[i], str(i))
            ix_list += [ix]
            iy_list += [iy]
        plt.axis("scaled")
    else:
        for i, iso_value in enumerate(np.linspace(0.0, 1.0, 1 + np.sum(n_isos))):
            # Get coeffs at given fixed point
            ix = x_interpolator(iso_value)
            iy = y_interpolator(iso_value)    
            ix_list += [ix]
            iy_list += [iy]
    return ix_list, iy_list


def get_intensity(pro, x, y, k=3):
    """
    Get mean intensity of all pixels in the kernel of size k on protein channel

    Args
    --------------------
    pro: np.array
        2D image of protein channel.
    x: list
        x coordinates of the kernel center
    y: list
        y coordinates of the kernel center
    k: int
        Kernel size, odd

    Returns
    -------
    matrix: np.array
        matrix of size [x,y] representing protein intensity representation at all input points
    """ 
    assert x.shape == y.shape
    shape = x.shape
    x = x.round().astype('uint16').flatten()
    y = y.round().astype('uint16').flatten()
    matrix = []
    for p in zip(x,y):
        k_c = kernel_coordinates(p, k=k)
        kernel_intensity = []
        for pi in k_c:
            if pi[0]<pro.shape[0] and pi[1]<pro.shape[1]:
                kernel_intensity += [pro[pi]]
        if len(kernel_intensity) == 0:
            kernel_intensity=[0]
        matrix += [np.mean(kernel_intensity)]
    matrix = np.array(matrix).reshape(shape)
    return matrix
'''
def get_intensity_interpolations(pc_bins, df, keep_list):
    """
    Get mean protein intensity of all interpolated points in each binned pc

    Args
    --------------------
    pro: np.array
        2D image of protein channel.
    x: list
        x coordinates of the kernel center
    y: list
        y coordinates of the kernel center
    k: int
        Kernel size, odd

    Returns
    -------
    matrix: np.array
        matrix of size [x,y] representing protein intensity representation at all input points
    """ 

    intensities_pcX = []
    for ls in pc_bins:
        intensities = []
        for l in ls:
            if l in keep_list:
                protein_path = Path(str(l).replace(".npy","_protein.png"))
                ori_fft = df.loc[df.index== l].values[0]
                shifts[l]
                
                intensity = plotting.get_protein_intensity(
                    pro_path = protein_path, 
                    shift_dict = shifts[link],
                    ori_fft = ori_fft, 
                    n_coef = n_coef, 
                    inverse_func = inverse_func
                    )
                intensities += [intensity.flatten()]
    
        if len(intensities) == 0:
            print('No cell sample at this bin for Nucleoplasm')
            intensities_pcX += [np.zeros_like(intensity)]
        else:
            print(len(intensities))
            intensities_pcX += [np.nanmean(intensities, axis=0).reshape(intensity.shape)]
    return intensities_pcX
'''