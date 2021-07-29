import numpy as np
import pandas as pd
from skimage.measure import find_contours, regionprops
from scipy.ndimage import center_of_mass, rotate
from utils import plotting, helpers, dimreduction, coefs
import matplotlib.pyplot as plt
import sys
import pathlib
from sklearn.decomposition import PCA


def align_cell_nuclei_centroids(data, plot=False):
    nuclei = data[1, :, :]
    cell = data[0, :, :]

    centroid_n = np.rint(center_of_mass(nuclei))
    centroid_c = np.rint(center_of_mass(cell))
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(nuclei, alpha=0.5)
        ax[0].imshow(cell, alpha=0.5)
        ax[0].plot(
            [centroid_c[1], centroid_n[1]], [centroid_c[0], centroid_n[0]], c="r"
        )
        ax[0].plot(
            [centroid_c[1] + 50, centroid_c[1]], [centroid_c[0], centroid_c[0]], c="b"
        )

    cn = [centroid_n[0] - centroid_c[0], centroid_n[1] - centroid_c[1]]
    c0 = [centroid_c[0] - centroid_c[0], centroid_c[1] + 50 - centroid_c[1]]
    theta = helpers.angle_between(cn, c0)
    if cn[0] < 0:
        theta = 360 - theta
    nuclei = rotate(nuclei, theta)
    cell = rotate(cell, theta)

    if plot:
        centroid_n = np.rint(center_of_mass(nuclei))
        centroid_c = np.rint(center_of_mass(cell))
        ax[1].imshow(nuclei, alpha=0.5)
        ax[1].imshow(cell, alpha=0.5)
        ax[1].plot(
            [centroid_c[1], centroid_n[1]], [centroid_c[0], centroid_n[0]], c="r"
        )
        ax[1].scatter(centroid_c[1], centroid_c[0])
        ax[1].set_title(f"rotate by {np.round(theta,2)}°")

    return nuclei, cell


def align_cell_major_axis(data, plot=True):
    nuclei = data[1, :, :]
    cell = data[0, :, :]
    region = regionprops(cell)[0]
    theta = region.orientation * 180 / np.pi  # radiant to degree conversion
    cell_ = rotate(cell, 90 - theta)
    nuclei_ = rotate(nuclei, 90 - theta)
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(nuclei, alpha=0.5)
        ax[0].imshow(cell, alpha=0.5)
        ax[1].imshow(nuclei_, alpha=0.5)
        ax[1].imshow(cell_, alpha=0.5)
    return nuclei_, cell_


def get_coefs_df(imlist, n_coef=32, func=None, plot=False):
    coef_df = pd.DataFrame()
    names = []
    error_n = []
    error_c = []
    for im in imlist:
        data = np.load(im)
        try:
            # nuclei, cell = align_cell_nuclei_centroids(data, plot=True)
            nuclei, cell = align_cell_major_axis(data, plot=False)

            centroid = center_of_mass(cell)
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
    return coef_df, names


get_coef_fun = coefs.fourier_coeffs  # coefs.wavelet_coefs  #
inverse_func = coefs.inverse_fft  # coefs.inverse_wavelet

d = pathlib.Path("C:/Users/trang.le/Desktop/2D_shape_space/U2OS")
imlist = [i for i in d.glob("*.npy")]
fourier_df = dict()
for n_coef in [128]:
    df_, names_ = get_coefs_df(imlist, n_coef, func=get_coef_fun)
    fourier_df[f"fourier_10rep_startalign_{n_coef}"] = df_
    df_.index = names_

n_coef = 128
df = fourier_df[f"fourier_10rep_startalign_{n_coef}"].copy()
use_complex = False
if get_coef_fun == coefs.fourier_coeffs:
    if not use_complex:
        df_ = pd.concat(
            [pd.DataFrame(np.matrix(df).real), pd.DataFrame(np.matrix(df).imag)], axis=1
        )
        pca = PCA()
        pca.fit(df_)
        plotting.display_scree_plot(pca)
    else:
        df_ = df
        pca = dimreduction.ComplexPCA(n_components=df_.shape[1])
        pca.fit(df_)
        plotting.display_scree_plot(pca)
elif get_coef_fun == coefs.wavelet_coefs:
    df_ = df
    pca = PCA(n_components=df_.shape[1])
    pca.fit(df_)
    plotting.display_scree_plot(pca)

matrix_of_features_transform = pca.transform(df_)
pc_names = [f"PC{c}" for c in range(1, 1 + len(pca.components_))]
pc_keep = [f"PC{c}" for c in range(1, 1 + 10)]
df_trans = pd.DataFrame(data=matrix_of_features_transform.copy())
df_trans.columns = pc_names
df_trans.index = df.index
df_trans[list(set(pc_names) - set(pc_keep))] = 0

if get_coef_fun == coefs.fourier_coeffs and not use_complex:
    df_sep_inv = pca.inverse_transform(df_trans)
    # df_inv = sc.inverse_transform(df_scaled_inv)

    real = df_sep_inv[:, : n_coef * 4]
    imag = df_sep_inv[:, n_coef * 4 :]
    cdf = []
    for s in range(len(real)):
        cdf.append([complex(r, i) for r, i in zip(real[s], imag[s])])
    df_inv = pd.DataFrame(np.matrix(cdf), index=df.index)
    # df_inv = df_scaled_inv
else:
    df_inv = pd.DataFrame(pca.inverse_transform(df_trans), index=df.index)

n_coef = df.shape[1] // 4
i = 0
for link, row in df_inv.iterrows():
    i = i + 1
    if i < 50:
        continue
    fcoef_c = row[0 : n_coef * 2]
    fcoef_n = row[n_coef * 2 :]
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(plt.imread(link.with_suffix(".jpg")))
    ori_fft = df.iloc[i - 1]
    for fcoef in [ori_fft[: n_coef * 2], ori_fft[n_coef * 2 :]]:
        ix__, iy__ = inverse_func(fcoef[:n_coef], fcoef[n_coef:])
        ax[1].scatter(ix__[0], iy__[0], color="r")
        ax[1].plot(ix__, iy__)
        ax[1].axis("scaled")
    for fcoef in [fcoef_c, fcoef_n]:
        ix_, iy_ = inverse_func(fcoef[:n_coef], fcoef[n_coef:])
        ax[2].scatter(ix_[0], iy_[0], color="r")
        ax[2].plot(ix_, iy_)
        ax[2].axis("scaled")
    if i > 70:
        breakme

midpoints = df_trans.clip(0, None).mean()
midpoints = df_trans.clip(None, 0).mean()

midpoints = df_trans.mean()
fcoef = pca.inverse_transform(midpoints)

if get_coef_fun == coefs.fourier_coeffs and use_complex:
    midpoints = []
    for c in df_trans:
        col = df_trans[c]
        real_ = [x.real for x in col]
        real = [-abs(x) for x in real_]

        imag_ = [x.imag for x in col]
        imag = imag_  # [abs(x) for x in imag_]
        # std += [complex(np.std(real), np.std(imag))]
        midpoints += [complex(np.mean(real), np.mean(imag))]
    # midpoints = df_trans.mean()
    fcoef = pca.inverse_transform(midpoints)

if not use_complex:
    real = fcoef[: len(fcoef) // 2]
    imag = fcoef[len(fcoef) // 2 :]
    fcoef = [complex(r, i) for r, i in zip(real, imag)]

# fcoef = df_inv.mean()
fcoef_c = fcoef[0 : n_coef * 2]
fcoef_n = fcoef[n_coef * 2 :]
ix_n, iy_n = inverse_func(fcoef_n[0:n_coef], fcoef_n[n_coef:])
ix_c, iy_c = inverse_func(fcoef_c[0:n_coef], fcoef_c[n_coef:])
plt.plot(ix_n, iy_n)
plt.plot(ix_c, iy_c)
plt.axis("scaled")

pm = plotting.PlotShapeModes(
    pca,
    df_trans,
    n_coef,
    pc_keep,
    scaler=None,
    complex_type=use_complex,
    inverse_func=inverse_func,
)
pm.plot_avg_cell()
for pc in pc_keep:
    pm.plot_shape_variation_gif(pc)
    pm.plot_pc_dist(pc)
    pm.plot_pc_hist(pc)
    pm.plot_shape_variation(pc)


# https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
# for each cell, do random start x100 and average the fft.
# Move back to complex numbers
# add beginning and end signals (10points eg) so the x, y is not periodic anymore
