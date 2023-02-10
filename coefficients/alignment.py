from lib2to3.pgen2.token import NT_OFFSET
import numpy as np
import pandas as pd
from skimage.measure import find_contours, regionprops
from scipy.ndimage import center_of_mass, rotate
from utils import helpers
import matplotlib.pyplot as plt
from pathlib import Path
from imageio import imread
import pickle
import os
import sys
sys.path.append("..") 
from coefficients.coefs import find_nearest, find_centroid

def align_cell_nuclei_centroids(data, protein_ch, plot=False):
    """
    Alignment of cells based on cell-nucleus centroid vector

    Parameters
    ----------
    data : npy array of size (2,x,y)
        numpy array of cell and nuclei masks.
    protein_ch : array of size (x,y)
        protein channels.
    plot : bool, optional
        Whether to plot the original and aligned masks. The default is False.

    Returns
    -------
    nuclei : 2D array 
        aligned nuclei.
    cell : 2D array 
        aligned cell.
    theta : float
        ° angle to rotate.

    """
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
        center_ = center_of_mass(nuclei_)
        ax[2].scatter(center_[1],center_[0])
        ax[3].imshow(protein_ch_)
    return nuclei_, cell_, 90-theta

def align_nuclei_major_axis(data, protein_ch, plot=True):
    nuclei = data[1, :, :]
    cell = data[0, :, :]
    region = regionprops(nuclei)[0]
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
        center_ = center_of_mass(nuclei_)
        ax[2].scatter(center_[1],center_[0])
        ax[3].imshow(protein_ch_)
    return nuclei_, cell_, 90-theta

def align_cell_major_axis_polarized(data, protein_ch, plot=True):
    nuclei = data[1, :, :]
    cell = data[0, :, :]
    region = regionprops(cell)[0]
    theta = region.orientation * (180 / np.pi)  # radiant to degree conversion
    theta = 90 - theta
    cell_ = rotate(cell, theta)
    nuclei_ = rotate(nuclei, theta)
    center_cell = center_of_mass(cell_)
    center_nuclei = center_of_mass(nuclei_)
    shape = nuclei_.shape
    
    # NOTE: np.rot90() flip counter-clockwise
    if center_cell[1] > center_nuclei[1]: # Move 1 quadrant counter-clockwise
        cell_ = rotate(cell_, 180)
        nuclei_ = rotate(nuclei_, 180)
        theta += 180
    
    theta = theta % 360 
    protein_ch_ = rotate(protein_ch, theta)
    if plot:
        center_ = center_of_mass(cell_)
        fig, ax = plt.subplots(1, 4, figsize=(8, 4))
        ax[0].imshow(nuclei, alpha=0.5)
        ax[0].imshow(cell, alpha=0.5)
        ax[1].imshow(protein_ch)
        ax[2].set_title(f"theta = {np.round(theta,1)}°")
        ax[2].imshow(nuclei_, alpha=0.5)
        ax[2].imshow(cell_, alpha=0.5)
        ax[2].scatter(center_[1],center_[0])
        ax[2].vlines(shape[1]//2, 0, shape[0]-1, colors='gray', linestyles ='dashed') 
        ax[2].hlines(shape[0]//2, 0, shape[1]-1, colors='gray', linestyles ='dashed') 
        ax[3].imshow(protein_ch_)
    return nuclei_, cell_, theta

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
            # nuclei_, cell_, theta = align_cell_nuclei_centroids(data, pro, plot=False)
            nuclei_, cell_, theta = align_cell_major_axis(data, pro, plot=plot)
            centroid = center_of_mass(nuclei_)
            # centroid = center_of_mass(cell)
            
            # Padd surrounding with 0 so no contour touch the border. This help matching squares algo not failing
            nuclei = np.zeros((nuclei_.shape[0]+2, nuclei_.shape[1]+2))
            nuclei[1:1+nuclei_.shape[0],1:1+nuclei_.shape[1]] = nuclei_
            cell = np.zeros((cell_.shape[0]+2, cell_.shape[1]+2))
            cell[1:1+cell_.shape[0],1:1+cell_.shape[1]] = cell_
            
            nuclei_coords_ = find_contours(nuclei, 0, fully_connected='high') 
            nuclei_coords_ = nuclei_coords_[0] - centroid

            cell_coords_ = find_contours(cell, 0, fully_connected='high') 
            if len(cell_contour) > 1:
                cell_contour = np.vstack(cell_contour)
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

def get_coefs_im(im, save_dir, log_dir, n_coef=32, func=None, plot=False):
    try:
        data = np.load(im)
    except:
        print(f"Check file size or format: {im}")
    pro = imread(Path(str(im).replace('.npy', '_protein.png')))
    try:
        # nuclei_, cell_, theta = align_cell_nuclei_centroids(data, pro, plot=False)
        # nuclei_, cell_, theta = align_cell_major_axis(data, pro, plot=False)
        nuclei_, cell_, theta = align_cell_major_axis_polarized(data, pro, plot=False)
        #centroid = center_of_mass(nuclei_)
        centroid = center_of_mass(nuclei_)
        
        # Padd surrounding with 0 so no contour touch the border. This help matching squares algo not failing (as much)
        nuclei = np.zeros((nuclei_.shape[0]+2, nuclei_.shape[1]+2))
        nuclei[1:1+nuclei_.shape[0],1:1+nuclei_.shape[1]] = nuclei_
        cell = np.zeros((cell_.shape[0]+2, cell_.shape[1]+2))
        cell[1:1+cell_.shape[0],1:1+cell_.shape[1]] = cell_
        
        nuclei_coords_ = find_contours(nuclei)
        if len(nuclei_coords_) > 1: # concatenate fragmented contour lines, original point could be ambiguous! (attempt to re-align original point in coefs.XXX_fourier_coefs())
            # if the biggest contour segment is a close loop, the rest are micronuclei, artifact segments
            idx_longest = np.argmax([len(xy) for xy in nuclei_coords_])
            biggest = nuclei_coords_[idx_longest]
            if all(biggest[0] == biggest[-1]):
                nuclei_coords_ = biggest - centroid
            else:
                nuclei_coords_ = np.vstack(nuclei_coords_)
                nuclei_coords_ = nuclei_coords_ - centroid
        else:
            nuclei_coords_ = nuclei_coords_[0] - centroid

        cell_coords_ = find_contours(cell)
        if len(cell_coords_) > 1: # concatenate fragmented contour lines, original point could be ambiguous! (attempt to re-align original point in coefs.XXX_fourier_coefs())
            # if the biggest contour segment is a close loop, the rest are micronuclei, artifact segments
            idx_longest = np.argmax([len(xy) for xy in cell_coords_])
            biggest = cell_coords_[idx_longest]
            if all(biggest[0] == biggest[-1]):
                cell_coords_ = biggest - centroid
            else:
                cell_coords_ = np.vstack(cell_coords_)
                cell_coords_ = cell_coords_ - centroid
        else:
            cell_coords_ = cell_coords_[0] - centroid

        cell_coords = cell_coords_.copy()
        cell_coords = helpers.realign_contour_startpoint(cell_coords)
        nuclei_coords = nuclei_coords_.copy()
        nuclei_coords = helpers.realign_contour_startpoint(nuclei_coords)
        if plot:
            fig, ax = plt.subplots(1, 3, figsize=(8, 4))
            ax[0].imshow(nuclei, alpha=0.5)
            ax[0].imshow(cell, alpha=0.5)
            
            nu_centroid = helpers.find_centroid(nuclei_coords)
            cell_centroid = helpers.find_centroid(cell_coords)
            ax[1].plot(nuclei_coords_[:, 0], nuclei_coords_[:, 1])
            ax[1].scatter(nuclei_coords_[0, 0], nuclei_coords_[0, 1], color="slateblue")
            ax[1].scatter(nu_centroid[0], nu_centroid[1], color="b")
            ax[1].plot(cell_coords_[:, 0], cell_coords_[:, 1])
            ax[1].scatter(cell_coords_[0, 0], cell_coords_[0, 1], color="gold")
            ax[1].scatter(cell_centroid[0], cell_centroid[1], color="orange")
            
            ax[1].vlines(0, -200, 200, colors='gray', linestyles ='dashed') 
            ax[1].hlines(0, -200, 200, colors='gray', linestyles ='dashed')
            ax[1].axis("scaled")
            ax[2].set_title(f"theta = {np.round(theta,1)}°")
            ax[2].vlines(0, -200, 200, colors='gray', linestyles ='dashed') 
            ax[2].hlines(0, -200, 200, colors='gray', linestyles ='dashed')
            ax[2].plot(nuclei_coords[:, 0], nuclei_coords[:, 1])
            ax[2].scatter(nuclei_coords[0, 0], nuclei_coords[0, 1], color="slateblue")
            ax[2].scatter(nu_centroid[0], nu_centroid[1], color="b")
            ax[2].plot(cell_coords[:, 0], cell_coords[:, 1])
            ax[2].scatter(cell_coords[0, 0], cell_coords[0, 1], color="gold")
            ax[2].scatter(cell_centroid[0], cell_centroid[1], color="orange")
            ax[2].axis("scaled")
            plt.savefig(f"{save_dir}/{os.path.basename(im)}.png", bbox_inches="tight")
            plt.close()

        fcoef_n, e_n = func(nuclei_coords, n=n_coef)
        fcoef_c, e_c = func(cell_coords, n=n_coef)
        #print(f"Saving to {save_dir}/fftcoefs_{n_coef}.txt")
        with open(f"{save_dir}/fftcoefs_{n_coef}.txt", "a") as F:
            F.write(",".join(map(str,[im]+np.concatenate([fcoef_c, fcoef_n]).ravel().tolist())) + '\n')

        with open(f"{save_dir}/shift_error_meta_fft{n_coef}.txt", "a") as F:
            # Saving: image_name, theta_alignment_rotation, shift_centroid, reconstruct_err_c, reconstruct_err_n
            F.write(";".join(map(str,[im, theta, centroid, e_c, e_n])) + '\n')
        return im, 1
    except:
        with open(f'{log_dir}/images_fft_failed.pkl', 'wb') as error_list:
            pickle.dump(f"Oops! {sys.exc_info()[0]} occurred for {im}", error_list)
        return im, 0

def get_coefs_nucleus(im, save_dir, log_dir, n_coef=32, func=None, plot=False):
    try:
        data = np.load(im)
    except:
        print(f"Check file size or format: {im}")
    pro = imread(Path(str(im).replace('.npy', '_protein.png')))
    try:
        nuclei_, cell_, theta = align_nuclei_major_axis(data, pro, plot=False)
        centroid = center_of_mass(nuclei_)
        
        # Padd surrounding with 0 so no contour touch the border. This help matching squares algo not failing (as much)
        nuclei = np.zeros((nuclei_.shape[0]+2, nuclei_.shape[1]+2))
        nuclei[1:1+nuclei_.shape[0],1:1+nuclei_.shape[1]] = nuclei_
        cell = np.zeros((cell_.shape[0]+2, cell_.shape[1]+2))
        cell[1:1+cell_.shape[0],1:1+cell_.shape[1]] = cell_
        
        nuclei_coords_ = find_contours(nuclei)
        if len(nuclei_coords_) > 1: # concatenate fragmented contour lines, original point could be ambiguous! (attempt to re-align original point in coefs.XXX_fourier_coefs())
            # if the biggest contour segment is a close loop, the rest are micronuclei, artifact segments
            idx_longest = np.argmax([len(xy) for xy in nuclei_coords_])
            biggest = nuclei_coords_[idx_longest]
            if all(biggest[0] == biggest[-1]):
                nuclei_coords_ = biggest - centroid
            else:
                nuclei_coords_ = np.vstack(nuclei_coords_)
                nuclei_coords_ = nuclei_coords_ - centroid
        else:
            nuclei_coords_ = nuclei_coords_[0] - centroid

        cell_coords_ = find_contours(cell)
        if len(cell_coords_) > 1: # concatenate fragmented contour lines, original point could be ambiguous! (attempt to re-align original point in coefs.XXX_fourier_coefs())
            # if the biggest contour segment is a close loop, the rest are micronuclei, artifact segments
            idx_longest = np.argmax([len(xy) for xy in cell_coords_])
            biggest = cell_coords_[idx_longest]
            if all(biggest[0] == biggest[-1]):
                cell_coords_ = biggest - centroid
            else:
                cell_coords_ = np.vstack(cell_coords_)
                cell_coords_ = cell_coords_ - centroid
        else:
            cell_coords_ = cell_coords_[0] - centroid

        cell_coords = cell_coords_.copy()
        cell_coords = helpers.realign_contour_startpoint(cell_coords)
        nuclei_coords = nuclei_coords_.copy()
        nuclei_coords = helpers.realign_contour_startpoint(nuclei_coords)
        if plot:
            fig, ax = plt.subplots(1, 3, figsize=(8, 4))
            ax[0].imshow(nuclei, alpha=0.5)
            ax[0].imshow(cell, alpha=0.5)
            
            nu_centroid = helpers.find_centroid(nuclei_coords)
            cell_centroid = helpers.find_centroid(cell_coords)
            ax[1].plot(nuclei_coords_[:, 0], nuclei_coords_[:, 1])
            ax[1].scatter(nuclei_coords_[0, 0], nuclei_coords_[0, 1], color="slateblue")
            ax[1].scatter(nu_centroid[0], nu_centroid[1], color="b")
            ax[1].plot(cell_coords_[:, 0], cell_coords_[:, 1])
            ax[1].scatter(cell_coords_[0, 0], cell_coords_[0, 1], color="gold")
            ax[1].scatter(cell_centroid[0], cell_centroid[1], color="orange")
            
            ax[1].vlines(0, -200, 200, colors='gray', linestyles ='dashed') 
            ax[1].hlines(0, -200, 200, colors='gray', linestyles ='dashed')
            ax[1].axis("scaled")
            ax[2].set_title(f"theta = {np.round(theta,1)}°")
            ax[2].vlines(0, -200, 200, colors='gray', linestyles ='dashed') 
            ax[2].hlines(0, -200, 200, colors='gray', linestyles ='dashed')
            ax[2].plot(nuclei_coords[:, 0], nuclei_coords[:, 1])
            ax[2].scatter(nuclei_coords[0, 0], nuclei_coords[0, 1], color="slateblue")
            ax[2].scatter(nu_centroid[0], nu_centroid[1], color="b")
            ax[2].plot(cell_coords[:, 0], cell_coords[:, 1])
            ax[2].scatter(cell_coords[0, 0], cell_coords[0, 1], color="gold")
            ax[2].scatter(cell_centroid[0], cell_centroid[1], color="orange")
            ax[2].axis("scaled")
            plt.savefig(f"{save_dir}/{os.path.basename(im)}.png", bbox_inches="tight")
            plt.close()

        fcoef_n, e_n = func(nuclei_coords, n=n_coef)
        fcoef_c, e_c = func(cell_coords, n=n_coef)
        
        with open(f"{save_dir}/fftcoefs_{n_coef}.txt", "a") as F:
            F.write(",".join(map(str,[im]+np.concatenate([fcoef_c, fcoef_n]).ravel().tolist())) + '\n')

        with open(f"{save_dir}/shift_error_meta_fft{n_coef}.txt", "a") as F:
            # Saving: image_name, theta_alignment_rotation, shift_centroid, reconstruct_err_c, reconstruct_err_n
            F.write(";".join(map(str,[im, theta, centroid, e_c, e_n])) + '\n')
        return im, 1
    except:
        with open(f'{log_dir}/images_fft_failed.pkl', 'wb') as error_list:
            pickle.dump(f"Oops! {sys.exc_info()[0]} occurred for {im}", error_list)
        return im, 0

# TO LOOK:
# https://stackoverflow.com/questions/59701966/forming-complex-number-array-in-python-from-test-file-of-two-column

