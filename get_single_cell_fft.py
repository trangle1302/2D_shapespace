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
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        ax[0].imshow(nuclei, alpha=0.5)
        ax[0].imshow(cell, alpha=0.5)
        ax[0].plot([centroid_c[1],centroid_n[1]],[centroid_c[0],centroid_n[0]], c ='r')
        ax[0].plot([centroid_c[1]+50,centroid_c[1]],[centroid_c[0],centroid_c[0]], c ='b')

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
        ax[1].plot([centroid_c[1],centroid_n[1]],[centroid_c[0],centroid_n[0]], c ='r')
        ax[1].scatter(centroid_c[1],centroid_c[0])
        ax[1].set_title(f'rotate by {np.round(theta,2)}°')
        
    return nuclei, cell

def align_cell_major_axis(data, plot=True):
    nuclei = data[1, :, :]
    cell = data[0, :, :]
    region = regionprops(cell)[0]
    theta=region.orientation*180/np.pi #radiant to degree conversion
    cell_ = rotate(cell, 90-theta)
    nuclei_ = rotate(nuclei, 90-theta)
    if plot:
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        ax[0].imshow(nuclei, alpha=0.5)
        ax[0].imshow(cell, alpha=0.5)
        ax[1].imshow(nuclei_, alpha=0.5)
        ax[1].imshow(cell_, alpha=0.5)
    return nuclei_, cell_

def get_coefs_df(imlist, n_coef=32, plot=False):
    fourier = pd.DataFrame()
    names = []
    error_n = []
    error_c = []
    for im in imlist:
        data = np.load(im)
        try:
            #nuclei, cell = align_cell_nuclei_centroids(data, plot=True)
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
                fig, ax = plt.subplots(1,3, figsize=(8,4))
                ax[0].imshow(nuclei, alpha=0.5)
                ax[0].imshow(cell, alpha=0.5)
                ax[1].plot(nuclei_coords_[:,0],nuclei_coords_[:,1])
                ax[1].plot(cell_coords_[:,0],cell_coords_[:,1])
                ax[1].axis('scaled')
                ax[2].plot(nuclei_coords[:,0],nuclei_coords[:,1])
                ax[2].plot(cell_coords[:,0],cell_coords[:,1])
                ax[2].axis('scaled')
                plt.show()
            
            fcoef_n, e_n = coefs.fourier_coeffs(nuclei_coords, n=n_coef)
            fcoef_c, e_c = coefs.fourier_coeffs(cell_coords, n=n_coef)

            error_c += [e_c]
            error_n += [e_n]
            fourier = fourier.append(
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
    return fourier, names


d = pathlib.Path("C:/Users/trang.le/Desktop/2D_shape_space/U2OS")
imlist = [i for i in d.glob("*.npy")]
fourier_df = dict()
names_df = dict()
for n_coef in [128,256]:
    df_, names_ = get_coefs_df(imlist, n_coef=n_coef)
    fourier_df[f"fft_cellmajoraxis_{n_coef}"] = df_
    names_df[f"fft_cellmajoraxis_{n_coef}"] = names_
    df_.index = names_
    df_.to_csv(
        f"C:/Users/trang.le/Desktop/2D_shape_space/tmp/fft_fftshift_vhflip_{n_coef}.csv"
    )

n_coef = 257
df = fourier_df["fft_hamm_10_shift_256"].copy()
'''
df_ = df
pca = dimreduction.ComplexPCA(n_components=df_.shape[1])
pca.fit(df_)
plotting.display_scree_plot(pca)
'''

df_ = pd.concat(
    [pd.DataFrame(np.matrix(df).real), pd.DataFrame(np.matrix(df).imag)], axis=1
)
pca2 = PCA()
pca2.fit(df_)
plotting.display_scree_plot(pca2)

matrix_of_features_transform = pca2.transform(df_)
pc_names = [f"PC{c}" for c in range(1, 1 + len(pca2.explained_variance_ratio_))]
pc_keep = [f"PC{c}" for c in range(1, 1 + 7)]
df_trans = pd.DataFrame(data=matrix_of_features_transform.copy())
df_trans.columns = pc_names
df_trans.index = df.index
df_trans[list(set(pc_names) - set(pc_keep))] = 0
df_sep_inv = pca2.inverse_transform(df_trans)
# df_inv = sc.inverse_transform(df_scaled_inv)

real = df_sep_inv[:, : n_coef * 4]
imag = df_sep_inv[:, n_coef * 4 :]
cdf = []
for s in range(len(real)):
    cdf.append([complex(r, i) for r, i in zip(real[s], imag[s])])
df_inv = pd.DataFrame(np.matrix(cdf), index=df.index)
# df_inv = df_scaled_inv

i = 0
for link, row in df_inv.iterrows():
    i = i + 1
    if i < 100:
        continue
    fcoef_c = row[0 : n_coef * 2]
    fcoef_n = row[n_coef * 2 :]
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(plt.imread(link.with_suffix(".jpg")))
    ori_fft = df.iloc[i - 1]
    for fcoef in [ori_fft[: n_coef * 2], ori_fft[n_coef * 2 :]]:
        ix__, iy__ = coefs.inverse_fft(fcoef[:n_coef], fcoef[n_coef:])
        ax[1].plot(ix__, iy__)
        ax[1].axis("scaled")
    for fcoef in [fcoef_c, fcoef_n]:
        ix_, iy_ = coefs.inverse_fft(fcoef[:n_coef], fcoef[n_coef:])
        ax[2].plot(ix_, iy_)
        ax[2].axis("scaled")
    if i > 110:
        breakme


# avg cell
tmp = df_trans.median(axis=0)
tmp_ = pca2.inverse_transform(tmp)
# tmp_ = sc.inverse_transform(tmp_)
real = tmp_[: len(tmp_) // 2]
imag = tmp_[len(tmp_) // 2 :]
tmp_ = [complex(r, i) for r, i in zip(real, imag)]
fcoef_c = tmp_[0 : n_coef * 2]
fcoef_n = tmp_[n_coef * 2 :]
ix_n, iy_n = coefs.inverse_fft(fcoef_n[:n_coef], fcoef_n[n_coef:],n=n_coef)
ix_c, iy_c = coefs.inverse_fft(fcoef_c[:n_coef], fcoef_c[n_coef:],n=n_coef)

fft_x = np.concatenate((fcoef_c[:n_coef], np.conjugate(fcoef_c[:n_coef][1:][::-1])))
fft_y = np.concatenate((fcoef_c[n_coef:], np.conjugate(fcoef_c[n_coef:][1:][::-1])))

fft_x = np.concatenate((fcoef_c[:n_coef], np.zeros((16 * n_coef)), np.conjugate(fcoef_c[:n_coef][1:][::-1]))
        )
fft_y = np.concatenate((fcoef_c[n_coef:], np.zeros((16 * n_coef)), np.conjugate(fcoef_c[n_coef:][1:][::-1]))
        )
        
ix = np.fft.ifft(fft_x)
iy = np.fft.ifft(fft_y)

ix = ix / np.hamming(len(ix))
iy = iy / np.hamming(len(iy))

ix_n, iy_n = coefs.equidistance(ix_n.real,iy_n.real, n_coef*5)
ix_c, iy_c = coefs.equidistance(ix_c.real,iy_c.real, n_coef*5)

plt.plot(ix_n, iy_n)
plt.plot(ix_c, iy_c)
plt.axis("scaled")

pm = plotting.PlotShapeModes(pca2, df_trans, n_coef, pc_keep, scaler=None)
for pc in pc_keep:
    pm.plot_shape_variation_gif(pc)
    pm.plot_pc_dist(pc)
    pm.plot_pc_hist(pc)
    pm.plot_shape_variation(pc)


# https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
