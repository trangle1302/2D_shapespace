import numpy as np
import pandas as pd
from skimage.measure import find_contours, regionprops
from scipy.ndimage import center_of_mass, rotate
from utils import helpers
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from imageio import imread

def align_cell_nuclei_centroids(data, protein_ch, plot=False):
    nuclei = data[1, :, :]
    cell = data[0, :, :]

    centroid_n = np.rint(center_of_mass(nuclei))
    centroid_c = np.rint(center_of_mass(cell))
    if plot:
        fig, ax = plt.subplots(1, 4, figsize=(8, 4))
        ax[0].imshow(nuclei, alpha=0.5)
        ax[0].imshow(cell, alpha=0.5)
        ax[0].plot(
            [centroid_c[1], centroid_n[1]], [centroid_c[0], centroid_n[0]], c="r"
        )
        ax[0].plot(
            [centroid_c[1] + 50, centroid_c[1]], [centroid_c[0], centroid_c[0]], c="b"
        )
        ax[1].imshow(protein_ch)

    cn = [centroid_n[0] - centroid_c[0], centroid_n[1] - centroid_c[1]]
    c0 = [centroid_c[0] - centroid_c[0], centroid_c[1] + 50 - centroid_c[1]]
    theta = helpers.angle_between(cn, c0)
    if cn[0] < 0:
        theta = 360 - theta
    nuclei = rotate(nuclei, theta)
    cell = rotate(cell, theta)
    protein_ch = rotate(protein_ch, theta)

    if plot:
        centroid_n = np.rint(center_of_mass(nuclei))
        centroid_c = np.rint(center_of_mass(cell))
        ax[2].imshow(nuclei, alpha=0.5)
        ax[2].imshow(cell, alpha=0.5)
        ax[2].plot(
            [centroid_c[1], centroid_n[1]], [centroid_c[0], centroid_n[0]], c="r"
        )
        ax[2].scatter(centroid_c[1], centroid_c[0])
        ax[2].set_title(f"rotate by {np.round(theta,2)}°")
        ax[3].imshow(protein_ch)

    return nuclei, cell, theta


def align_cell_major_axis(data, protein_ch, plot=True):
    nuclei = data[1, :, :]
    cell = data[0, :, :]
    region = regionprops(cell)[0]
    theta = region.orientation * 180 / np.pi  # radiant to degree conversion
    cell_ = rotate(cell, 90 - theta)
    nuclei_ = rotate(nuclei, 90 - theta)
    protein_ch_ = rotate(protein_ch, 90-theta)
    if plot:
        fig, ax = plt.subplots(1, 4, figsize=(8, 4))
        ax[0].imshow(nuclei, alpha=0.5)
        ax[0].imshow(cell, alpha=0.5)
        ax[1].imshow(protein_ch)
        ax[2].imshow(nuclei_, alpha=0.5)
        ax[2].imshow(cell_, alpha=0.5)
        ax[3].imshow(protein_ch_)
    return nuclei_, cell_, 90-theta


def get_coefs_df(imlist, n_coef=32, func=None, plot=False):
    coef_df = pd.DataFrame()
    shifts = dict()
    names = []
    error_n = []
    error_c = []
    for im in imlist:
        data = np.load(im)
        pro = imread(Path(str(im).replace('.npy', '_protein.png')))
        try:
            nuclei, cell, theta = align_cell_nuclei_centroids(data, pro, plot=False)
            # nuclei, cell = align_cell_major_axis(data, pro, plot=False)
            centroid = center_of_mass(nuclei)
            # centroid = center_of_mass(cell)
            nuclei_coords_ = find_contours(nuclei)
            nuclei_coords_ = nuclei_coords_[0] - centroid

            cell_coords_ = find_contours(cell)
            cell_coords_ = cell_coords_[0] - centroid

            if min(cell_coords_[:, 0]) > 0 or min(cell_coords_[:, 1]) > 0:
                print(f"Contour failed {im}")
                continue
            elif max(cell_coords_[:, 0]) < 0 or max(cell_coords_[:, 1]) < 0:
                print(f"Contour failed {im}")
                continue
            shifts.update({im: {"theta": theta, "shift_c": centroid}})
            cell_coords = cell_coords_.copy()
            nuclei_coords = nuclei_coords_.copy()
            if plot:
                fig, ax = plt.subplots(1, 3, figsize=(8, 4))
                ax[0].imshow(nuclei, alpha=0.5)
                ax[0].imshow(cell, alpha=0.5)
                ax[1].plot(nuclei_coords_[:, 0], nuclei_coords_[:, 1])
                ax[1].plot(cell_coords_[:, 0], cell_coords_[:, 1])
                ax[1].axis("scaled")
                ax[2].plot(nuclei_coords[:, 0], nuclei_coords[:, 1])
                ax[2].plot(cell_coords[:, 0], cell_coords[:, 1])
                ax[2].scatter(cell_coords[0, 0], cell_coords[0, 1], color="r")
                ax[2].axis("scaled")
                plt.show()

            fcoef_n, e_n = func(nuclei_coords, n=n_coef)
            fcoef_c, e_c = func(cell_coords, n=n_coef)

            error_c += [e_c]
            error_n += [e_n]
            coef_df = coef_df.append(
                [np.concatenate([fcoef_c, fcoef_n]).ravel().tolist()], ignore_index=True
            )
            names += [im]
        except:
            print("Oops!", sys.exc_info()[0], "occurred.")
            print(im)
            continue
    print(f"Get coefficients for {len(names)}/{len(imlist)} cells")
    print(f"Reconstruction error for nucleus: {np.average(error_n)}")
    print(f"Reconstruction error for cell: {np.average(error_c)}")
    return coef_df, names, shifts
