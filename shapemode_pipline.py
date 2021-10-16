from utils.parameterize import get_coordinates
from utils import plotting, helpers, dimreduction, coefs, alignment
from sklearn.decomposition import PCA
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Coefficients
fun = "fft"
if fun == "fft":
    get_coef_fun = coefs.fourier_coeffs  # coefs.wavelet_coefs  #
    inverse_func = coefs.inverse_fft  # coefs.inverse_wavelet
elif fun == "wavelet":
    get_coef_fun = coefs.wavelet_coefs
    inverse_func = coefs.inverse_wavelet

d = Path("C:/Users/trang.le/Desktop/2D_shape_space/U2OS_2")
imlist = [i for i in d.glob("*.npy")]
fourier_df = dict()
for n_coef in [128]:
    df_, names_, shifts = alignment.get_coefs_df(imlist, n_coef, func=get_coef_fun)
    fourier_df[f"fourier_ccentroid_fft_{n_coef}"] = df_
    df_.index = names_

#%% PCA and shape modes
n_coef = 128
df = fourier_df[f"fourier_ccentroid_fft_{n_coef}"].copy()
use_complex = False
if fun == "fft":
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
elif fun == "wavelet":
    df_ = df
    pca = PCA(n_components=df_.shape[1])
    pca.fit(df_)
    plotting.display_scree_plot(pca)

matrix_of_features_transform = pca.transform(df_)
pc_names = [f"PC{c}" for c in range(1, 1 + len(pca.components_))]
pc_keep = [f"PC{c}" for c in range(1, 1 + 13)]
df_trans = pd.DataFrame(data=matrix_of_features_transform.copy())
df_trans.columns = pc_names
df_trans.index = df.index
df_trans[list(set(pc_names) - set(pc_keep))] = 0


#%% Plotting
if fun == "fft" and not use_complex:
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
    if i < 10:
        continue
    shape_path = link.with_suffix(".png")
    protein_path = Path(str(link).replace(".npy","_protein.png"))
    ori_fft = df.iloc[i - 1]
    pca_fft = row
    save_path = Path("C:/Users/trang.le/Desktop/2D_shape_space/interpolations_plots").joinpath(shape_path.name)
    """
    plotting.plot_interpolations(shape_path = shape_path, 
                                 pro_path = protein_path,
                                 shift_dict = shifts[link],
                                 save_path = save_path,
                                 ori_fft = ori_fft, 
                                 reduced_fft = pca_fft, 
                                 n_coef = n_coef, 
                                 inverse_func = inverse_func)
    """
    plotting.plot_interpolation3(shape_path = shape_path, 
                                 pro_path = protein_path,
                                 shift_dict = shifts[link],
                                 save_path = save_path,
                                 ori_fft = ori_fft, 
                                 reduced_fft = pca_fft, 
                                 n_coef = n_coef, 
                                 inverse_func = inverse_func)
    if i > 25:
        breakme

midpoints = df_trans.clip(0, None).mean()
midpoints = df_trans.clip(None, 0).mean()

midpoints = df_trans.mean()
fcoef = pca.inverse_transform(midpoints)

if fun == "fft" and use_complex:
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

if fun == "fft" and not use_complex:
    real = fcoef[: len(fcoef) // 2]
    imag = fcoef[len(fcoef) // 2 :]
    fcoef = [complex(r, i) for r, i in zip(real, imag)]

# fcoef = df_inv.mean()
fcoef_c = fcoef[0 : n_coef * 2]
fcoef_n = fcoef[n_coef * 2 :]
ix_n, iy_n = inverse_func(fcoef_n[0:n_coef], fcoef_n[n_coef:])
ix_c, iy_c = inverse_func(fcoef_c[0:n_coef], fcoef_c[n_coef:])
plt.plot(ix_n.real, iy_n.real)
plt.plot(ix_c.real, iy_c.real)
plt.axis("scaled")
x_,y_ = get_coordinates(np.concatenate([ix_n.real, iy_n.real]), np.concatenate([ix_c.real, iy_c.real]), [0,0], n_isos = [5,5], plot=True)

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

"""
coeffs_mem = [(x.real, y.real) for x, y in zip(fcoef_c[0:n_coef], fcoef_c[n_coef:])]
coeffs_nuc = [(x.real, y.real) for x, y in zip(fcoef_n[0:n_coef], fcoef_n[n_coef:])]
centroid_nuc = helpers.find_centroid([(x.real, y.real) for x, y in zip(ix_n, iy_n)])
centroid_mem = helpers.find_centroid([(x.real, y.real) for x, y in zip(ix_c, iy_c)])
plt.plot(ix_n.real, iy_n.real)
plt.scatter(centroid_nuc[0],centroid_nuc[1], c="r")
plt.plot(ix_c.real, iy_c.real)
plt.scatter(centroid_mem[0],centroid_mem[1], c="b")
plt.axis("scaled")

# https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
y = np.array(x_) + shift_dict["shift_c"][0]
x = np.array(y_) + shift_dict["shift_c"][1]

m = get_intensity(protein_ch, x, y, k=3)
    
    

    x_n, y_n = ix_n.real, iy_n.real
    x_c, y_c = ix_c.real, iy_c.real
    iso_values = [0.0] + n_isos
    iso_values = np.cumsum(iso_values)
    iso_values = iso_values / iso_values[-1]
    
    
    x = np.c_[np.full_like(x_n, centroid[0]), x_n, x_c].ravel()
    y = np.c_[np.full_like(y_n, centroid[1]), y_n, y_c].ravel()
    
    points = np.array([(xi,yi) for xi, yi in zip(x,y)])
    z = np.hypot(x, y)
        
    xnew = np.append(np.linspace(x.min(), centroid[0], 8),np.linspace(centroid[0], x.max(), 3))        
    ynew = np.append(np.linspace(y.min(), centroid[1], 8),np.linspace(centroid[1], y.max(), 3))
    #ynew = np.linspace(0, 1, 1 + np.sum(n_isos)) #np.linspace(y.min(), y.max(), 1 + np.sum(n_isos))
    grid_z = scipy.interpolate.griddata((x,y), z, (xnew[None,:], ynew[:,None]), method='linear')
    
plt.plot(ix_n.real, iy_n.real)
plt.plot(ix_c.real, iy_c.real)
plt.contourf(xnew, ynew, grid_z, alpha=0.5)
plt.axis("scaled")
plt.show()

    
# interpolate to a grid
grid_x, grid_y = np.mgrid[min(x):max(x):(max(x)-min(x))/n_coef, min(y):max(y):(max(y)-min(y))/n_coef] 
grid_x, grid_y = np.mgrid[min(x):max(x):21j,min(y):max(y):21j]
grid_z = scipy.interpolate.griddata((x,y), z, (grid_x, grid_y), method='cubic')

plt.plot(ix_n.real, iy_n.real)
plt.plot(ix_c.real, iy_c.real)
plt.contourf(grid_x, grid_y, grid_z, alpha=0.5)
plt.axis("scaled")
plt.show()

levels = np.linspace(0.0, 1.0, 1 + np.sum(n_isos))
plt.ylabel('Y', size=15)
plt.xlabel('X', size=15)
cmap = plt.cm.jet_r
cs = plt.contourf(xnew, ynew, grid_linear, levels=levels, cmap=cmap)
cbar = plt.colorbar(cs)
cbar.set_label('Z', rotation=90, fontsize=15) # gas fraction
plt.show()


# Test data
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed
npts = 200
x = uniform(-2,2,npts)
y = uniform(-2,2,npts)
z = x*np.exp(-x**2-y**2)
# define grid.
xi = np.linspace(-2.1,2.1,100)
yi = np.linspace(-2.1,2.1,100)
# grid the data.
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
# contour the gridded data, plotting dots at the randomly spaced data points.
CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
plt.scatter(x,y,marker='o',c='b',s=5)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.title('griddata test (%d points)' % npts)
plt.show()

"""
def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

#%%
df = pd.read_csv("C:/Users/trang.le/Desktop/annotation-tool//final_labels_allversions.csv")
df = df[df.atlas_name == "U-2 OS"]

df.sc_locations_reindex
df_inv.index

mappings = pd.DataFrame(df_inv.index, columns=['Link'])
mappings["basename"] = [l.stem for l in mappings.Link]
mappings["image_id"] = [n[:-2] for n in mappings.basename]
mappings["cell_id"] = [n.split("_")[-1] for n in mappings.basename]
mappings["cell_id"] = mappings.image_id + "/" + mappings.cell_id
mappings = mappings.merge(df, how='inner', on=["image_id","cell_id"])


# std normalization 
# keep 1st - 99th percentile, rm outliers
# check the histogram distribution of cells in each PC
# For each bin (0.5 std step), average the protein representations of all cells in the same bin

# For visualization: do nearest-neighbor interpolation on the mapped protein representation.