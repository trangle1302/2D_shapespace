import numpy as np
import pandas as pd
from skimage.measure import find_contours
from scipy.ndimage import center_of_mass
#from utils import *
from utils import plotting, helpers, dimreduction, coefs
import matplotlib.pyplot as plt
import sys
import pathlib


def get_coefs_df(imlist, n_coef=32):
    fourier = pd.DataFrame()
    names = []
    error_n = []
    error_c = []
    for im in imlist:
        data = np.load(im)
        try:
            nuclei = data[1, :, :]
            cell = data[0, :, :]
            centroid = np.rint(center_of_mass(nuclei)).astype('uint16')

            if cell[:,:centroid[1]].sum() < cell[:,centroid[1]:].sum():
                cell = np.fliplr(cell)
                nuclei = np.fliplr(nuclei)
            
            if cell[:centroid[0],:].sum() < cell[centroid[0]:,:].sum():
                cell = np.flipud(cell)
                nuclei = np.flipud(nuclei)         
            
            centroid = center_of_mass(nuclei)
            nuclei_coords_ = find_contours(nuclei,0.2)            
            nuclei_coords_ = nuclei_coords_[0] - centroid
            
            cell_coords_ = find_contours(cell)
            cell_coords_ = cell_coords_[0] - centroid

            if min(nuclei_coords_[:,0])>0 or min(nuclei_coords_[:,1])>0 or min(cell_coords_[:,0])>0 or min(cell_coords_[:,1])>0:
                print(f'Contour failed {im}')
                continue
            elif max(nuclei_coords_[:,0])<0 or max(nuclei_coords_[:,1])<0 or max(cell_coords_[:,0])<0 or max(cell_coords_[:,1])<0:
                print(f'Contour failed {im}')
                continue
            
            
            cell_coords = cell_coords_.copy()
            nuclei_coords = nuclei_coords_.copy()
            '''
            if sum(cell_coords[:,0]) < 0: #simple flipping, not correct for very concaved cells
                #print(f'flip:{im}')
                nuclei_coords[:,0] = 0-nuclei_coords[:,0]
                cell_coords[:,0] = 0-cell_coords[:,0]
            
            if sum(cell_coords[:,1]) < 0: #simple flipping, not correct for very concaved cells
                #print(f'flip:{im}')
                nuclei_coords[:,1] = 0-nuclei_coords[:,1]
                cell_coords[:,1] = 0-cell_coords[:,1]
            
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
            '''
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
for n_coef in [256, 512]:
    df_, names_ = get_coefs_df(imlist, n_coef=n_coef)
    fourier_df[f"fft_{n_coef}"] = df_
    names_df[f"fft_{n_coef}"] = names_
    df_.index = names_
    df_.to_csv(f"C:/Users/trang.le/Desktop/2D_shape_space/tmp/fft_fftshift_vhflip_{n_coef}.csv")

n_coef = 256
#df = pd.read_csv(f"C:/Users/trang.le/Desktop/2D_shape_space/tmp/fft_vhflip_{n_coef}.csv", index_col=0)
df = fourier_df["fft_256"].copy()
'''
magnitude = []
angle = []
for c in df:
    m, a = helpers.R2P(df[c])
    magnitude += [m]
    angle += [a]

sc = StandardScaler()
sc.fit(np.column_stack(magnitude))
magnitudes = sc.transform(np.column_stack(magnitude))
df_ = np.hstack([magnitudes,np.column_stack(angle)])
'''
df_=df#[:,:2*n_coef]
pca = dimreduction.ComplexPCA(n_components=df_.shape[1])
pca.fit(df_)
plotting.display_scree_plot(pca)

matrix_of_features_transform = pca.transform(df_)
pc_names = [f"PC{c}" for c in range(1, 1 + len(pca.explained_variance_ratio_))]
#matrix_of_features_transform.columns = pc_names
pc_keep = [f"PC{c}" for c in range(1, 1 + 12)]
df_trans = pd.DataFrame(data=matrix_of_features_transform.copy())
df_trans.columns = pc_names
df_trans[list(set(pc_names) - set(pc_keep))] = 0
df_scaled_inv = pca.inverse_transform(df_trans)
#df_inv= sc.inverse_transform(df_scaled_inv)

df_inv = df_scaled_inv  #np.exp(df_scaled_inv)

i=0
for link, row in df_inv.iterrows():
    i = i +1
    if i < 150:
       continue
    fcoef_c = row[0 : n_coef * 2]
    fcoef_n = row[n_coef * 2 :]
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(plt.imread(link.with_suffix(".jpg")))
    for fcoef in [fcoef_c, fcoef_n]:
        ix_, iy_ = coefs.inverse_fft(fcoef[0:n_coef], fcoef[n_coef:])
        ax[1].plot(ix_, iy_)
        ax[1].axis("scaled")
    if i > 160:
        breakme



# avg cell
tmp = df_trans.median(axis=0)
#tmp = df_trans.iloc[10]
tmp_ = pca.inverse_transform(tmp)
fcoef_c = tmp_[0:n_coef*2]
fcoef_n = tmp_[n_coef*2:]
ix_n, iy_n = coefs.inverse_fft(fcoef_n[0:n_coef], fcoef_n[n_coef:])
ix_c, iy_c = coefs.inverse_fft(fcoef_c[0:n_coef], fcoef_c[n_coef:])

#ix_n, iy_n = inverse_fft(tmp_[0:n_coef], tmp_[2*n_coef:3*n_coef])
#ix_c, iy_c = inverse_fft(tmp_[n_coef:2*n_coef], tmp_[3*n_coef:])

#ix_n, iy_n = equidistance(ix_n.real, iy_n.real, n_coef*10)
#ix_c, iy_c = equidistance(ix_c.real, iy_c.real, n_coef*10)
plt.plot(ix_n, iy_n)    
plt.plot(ix_c, iy_c)
plt.axis('scaled')

pm = plotting.PlotShapeModes(pca, df_trans, n_coef, pc_keep, scaler = None)
for pc in pc_keep:
    pm.plot_shape_variation_gif(pc)
    pm.plot_pc_dist(pc)
    pm.plot_pc_hist(pc)
    pm.plot_shape_variation(pc)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca