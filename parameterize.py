import os
import numpy as np
from scipy import interpolate as spinterp
from typing import Optional, List, Dict, Tuple
import warnings


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


def get_mapping_coordinates():

    return coords


def cellular_mapping():

    return mapping


def morph_representation_on_shape():
    return img
