import os
from coefs import alignment, coefs
from warps.parameterize import get_coordinates
from utils import plotting, helpers, dimreduction
from sklearn.decomposition import PCA
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import get_location_counts, interpolate_
import argparse
import imageio

LABEL_NAMES = {
  0: 'Nucleoplasm',
  1: 'Nuclear membrane',
  2: 'Nucleoli',
  3: 'Nucleoli fibrillar center',
  4: 'Nuclear speckles',
  5: 'Nuclear bodies',
  6: 'Endoplasmic reticulum',
  7: 'Golgi apparatus',
  8: 'Intermediate filaments',
  9: 'Actin filaments',
  10: 'Microtubules',
  11: 'Mitotic spindle',
  12: 'Centrosome',
  13: 'Plasma membrane',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'Vesicles and punctate cytosolic patterns',
  18: 'Negative',
}

all_locations = dict((v, k) for k,v in LABEL_NAMES.items())
#%% Coefficients
fun = "fft"
if fun == "fft":
    get_coef_fun = coefs.fourier_coeffs  # coefs.wavelet_coefs  #
    inverse_func = coefs.inverse_fft  # coefs.inverse_wavelet
elif fun == "wavelet":
    get_coef_fun = coefs.wavelet_coefs
    inverse_func = coefs.inverse_wavelet

d = Path("C:/Users/trang.le/Desktop/2D_shape_space/U2OS")
meta = pd.read_csv("C:/Users/trang.le/Desktop/annotation-tool//final_labels_allversions.csv")

imlist = [i for i in d.glob("*.npy")]
fourier_df = dict()
for n_coef in [128]:
    df_, names_, shifts = alignment.get_coefs_df(imlist, n_coef, func=get_coef_fun)
    fourier_df[f"fourier_ccentroid_fft_{n_coef}_fixed"] = df_
    df_.index = names_

df_.columns = [f'coef{i}' for i in range(len(df_.columns))]
save_path = os.path.join(d.cwd(),"fft",f"fourier_ccentroid_fft_{n_coef}.txt")
df_.to_csv(save_path)
df = pd.read_csv(save_path, index_col=0)
df = df.applymap(lambda s: np.complex(s.replace('i', 'j'))) 
"""
compare = (df_ == df)      # Dataframe of True/False
compare.all()              # By column, True if all values are equal
compare.count()            # By column, how many values are equal

# Return any rows where there was a difference
df.where(~compare).dropna(how='all')
"""
#%% PCA and shape modes
n_coef = 128
#df = fourier_df[f"fourier_ccentroid_fft_{n_coef}_fixed"].copy()
#df = df[df.index.isin(mappings.Link)]
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
pc_keep = [f"PC{c}" for c in range(1, 1 + 12)]
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
    shape_path = Path(link).with_suffix(".png")
    protein_path = Path(str(link).replace(".npy","_protein.png"))
    ori_fft = df.loc[df.index== link].values[0]
    pca_fft = row
    save_path = Path("C:/Users/trang.le/Desktop/2D_shape_space/interpolations_plots").joinpath(shape_path.name)
    """
    plotting.plot_interpolation2(shape_path = shape_path, 
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
                                 shift_dict = shifts[Path(link)],
                                 save_path = save_path,
                                 ori_fft = ori_fft, 
                                 reduced_fft = pca_fft, 
                                 n_coef = n_coef, 
                                 inverse_func = inverse_func)
    
    #if i > 508:
    #    breakme

#%%
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
x_,y_ = get_coordinates(np.concatenate([ix_n.real, iy_n.real]), np.concatenate([ix_c.real, iy_c.real]), [0,0], n_isos = [10,10], plot=True)

for i in range(10):
    fig, ax = plt.subplots()
    for i,(xi,yi,intensities) in enumerate(zip(x_,y_,intensities__pc1[i])):
        ax.scatter(xi, yi,c=intensities)
    ax.axis("scaled")
    ax.set_facecolor('#541352FF')
#%%
pm = plotting.PlotShapeModes(
    pca,
    df_trans,
    n_coef,
    pc_keep,
    scaler=None,
    complex_type=use_complex,
    inverse_func=inverse_func,
)
pm.plot_avg_cell(dark=False)
for pc in pc_keep:
    pm.plot_shape_variation_gif(pc, dark=False)
    pm.plot_pc_dist(pc)
    pm.plot_pc_hist(pc)
    pm.plot_shape_variation(pc, dark=False)

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
base_dir = "C:/Users/trang.le/Desktop/annotation-tool"
df_test = pd.read_csv(base_dir + "/final_labels_allversions_current_withv6.csv")
labels = pd.read_csv(base_dir + "/HPA-Challenge-2020-all/data_for_Kaggle/labels.csv")
labels["Image_ID"] = [l.split("_")[0] for l in labels.ID] 
labels["cell_ID"] = [str(int(l.split("_")[1]) - 1) for l in labels.ID] 
mappings = pd.read_csv(base_dir + "/HPA-Challenge-2020-all/mappings.csv")
labels = pd.merge(labels, mappings, on='Image_ID')
labels["cell_id"] = labels["HPA_ID"] + '/' +  labels["cell_ID"]
meta = pd.merge(labels, df_test, on = 'cell_id')
meta["HPA_ID"] = [os.path.basename(f) for f in meta.HPA_path]
ifimages = pd.read_csv("C:/Users/trang.le/Desktop/annotation-tool/HPA-Challenge-2020-all/IF-image_v21.csv")
ifimages["HPA_ID"] = [os.path.basename(f)[:-1] for f in ifimages.filename]
meta = meta.merge(ifimages, how="left", on = "HPA_ID")

mappings = pd.DataFrame(df.index, columns=['Link'])
mappings["basename"] = [l.stem for l in mappings.Link]
mappings["image_id"] = [n.rsplit("_",1)[0] for n in mappings.basename]
mappings["cell_id"] = [n.rsplit("_",1)[1] for n in mappings.basename]
mappings["cell_id"] = mappings.image_id + "/" + mappings.cell_id
mappings = mappings.merge(meta, how='inner', on=["image_id","cell_id"])
location_counts = get_location_counts(list(mappings.sc_locations_reindex), all_locations)

Gene = "P2RX1"
df_sl_Label = mappings[mappings.gene_names == Gene]
df_sl_Label.WindowLink = [Path(l) for l in df_sl_Label.Link]

LABELNAME = 'Nucleoplasm'
LABELINDEX = str(all_locations[LABELNAME])

for PC in pc_keep:
#df_sl_Label = mappings[mappings.sc_locations_reindex == LABELINDEX]
    pc1, pc1l = pm.assign_cells(PC)
    
    #pc1l_Nucleoplasm = [l for l in ls for ls in pc1l if l in df_sl_Nucleoplasm.Link]
    
    shape = (21,256)
    intensities__pc1 = []
    counts = []
    for ls in pc1l:
        intensities = []
        i= 0
        for l in ls:
            if l in list(df_sl_Label.Link):
                #print(l)
                protein_path = Path(str(l).replace(".npy","_protein.png"))
                ori_fft = df.loc[df.index== l].values[0]
                """
                fig, ax = plt.subplots()
                p = rotate(imread(protein_path), shifts[l]["theta"])
                #thresh = threshold_mean(p)
                p = exposure.equalize_hist(p)
                #p[p<thresh] = 0
                plt.imshow(p)    
                """
                intensity = plotting.get_protein_intensity(
                    pro_path = protein_path, 
                    shift_dict = shifts[l],
                    ori_fft = ori_fft, 
                    n_coef = n_coef, 
                    inverse_func = inverse_func
                    )
                
                #fig, ax = plt.subplots()
                #plt.imshow(intensity)
                intensities += [intensity.flatten()]
                i +=1
        counts += [i]
        if len(intensities) == 0:
            print('No cell sample at this bin for Nucleoplasm')
            intensities__pc1 += [np.zeros(shape)]
        else:
            print(len(intensities))
            intensities__pc1 += [np.nanmean(intensities, axis=0).reshape(intensity.shape)]
    
    pm.protein_intensities = intensities__pc1/np.array(intensities__pc1).max()
    pm.plot_protein_through_shape_variation_gif(PC)


from scipy.ndimage import rotate
from skimage import exposure
from imageio import imread
encoded_image_dir = 'C:/Users/trang.le/Desktop/annotation-tool/HPA-Challenge-2020-all/HPA_Kaggle_Challenge_2020/data_for_Kaggle/data'

for ls in pc1l:
    ls = pc1l[8]
    for l in ls:
        if l in list(df_sl_Label.Link):
            #print(l.name)
            encoded_image_id= mappings[mappings.image_id==l.name.rsplit('_',1)[0]].Image_ID.values[0]
            nu = imread(encoded_image_dir +'/' + encoded_image_id + '_blue.png')
            mt = imread(encoded_image_dir +'/' + encoded_image_id + '_red.png')
            protein = imread(encoded_image_dir +'/' + encoded_image_id + '_green.png')
            if mt.dtype == 'uint8':
                cell = np.dstack([mt, protein, nu])
            else:
                cell = (np.dstack([mt, protein, nu])/255).astype('uint8')
            protein_path = Path(str(l).replace(".npy","_protein.png"))
            fig, ax = plt.subplots(1,3)
            p = rotate(imread(protein_path), shifts[l]["theta"])
            
            mask = rotate(imread(Path(str(l).replace(".npy",".png"))), shifts[l]["theta"])
            #thresh = threshold_mean(p)
            #p = exposure.equalize_hist(p)
            #p[p<thresh] = 0
            ax[0].imshow(cell) 
            ax[0].set_axis_off()
            ax[1].imshow(p) 
            ax[1].set_axis_off()
            ax[2].imshow(mask) 
            ax[2].set_axis_off()
    
"""
for i, row in df_sl_Label.iterrows():
    l = row.Link
    protein_path = Path(str(l).replace(".npy","_protein.png"))
    fig, ax = plt.subplots()
    plt.imshow()    
"""
# std normalization 
# keep 1st - 99th percentile, rm outliers
# check the histogram distribution of cells in each PC
# For each bin (0.5 std step), average the protein representations of all cells in the same bin

# For visualization: do nearest-neighbor interpolation on the mapped protein representation.

#%% Map single organelles
for org in all_locations.keys():
    LABELINDEX = str(all_locations[org])
    df_sl_Label = mappings[mappings.sc_locations_reindex == LABELINDEX]
    
    for PC in pc_keep:
        pc1, pc1l = pm.assign_cells(PC)
                
        shape = (21,256)
        intensities__pc1 = []
        counts = []
        for ls in pc1l:
            intensities = []
            i= 0
            for l in ls:
                if l in list(df_sl_Label.Link):
                    #print(l)
                    protein_path = Path(str(l).replace(".npy","_protein.png"))
                    ori_fft = df.loc[df.index== l].values[0]
                    """
                    fig, ax = plt.subplots()
                    p = rotate(imread(protein_path), shifts[l]["theta"])
                    #thresh = threshold_mean(p)
                    p = exposure.equalize_hist(p)
                    #p[p<thresh] = 0
                    plt.imshow(p)    
                    """
                    intensity = plotting.get_protein_intensity(
                        pro_path = protein_path, 
                        shift_dict = shifts[l],
                        ori_fft = ori_fft, 
                        n_coef = n_coef, 
                        inverse_func = inverse_func
                        )
                    
                    #fig, ax = plt.subplots()
                    #plt.imshow(intensity)
                    intensities += [intensity.flatten()]
                    i +=1
            counts += [i]
            if len(intensities) == 0:
                print('No cell sample at this bin for Nucleoplasm')
                intensities__pc1 += [np.zeros(shape)]
            else:
                print(len(intensities))
                intensities__pc1 += [np.nanmean(intensities, axis=0).reshape(intensity.shape)]
        
        pm.protein_intensities = intensities__pc1/np.array(intensities__pc1).max()
        pm.plot_protein_through_shape_variation_gif(PC, title=org)

#%% Average intensity on average cells 
# plot on avg cells

x_,y_ = get_coordinates(np.concatenate([ix_n.real, iy_n.real]), np.concatenate([ix_c.real, iy_c.real]), [0,0], n_isos = [10,10], plot=True)
norm = plt.Normalize(vmin=0, vmax=1)

for org in list(all_locations.keys())[:-1]:
    LABELINDEX = str(all_locations[org])
    df_sl_Label = mappings[mappings.sc_locations_reindex == LABELINDEX]
    print(f'{org}, # of cells: {df_sl_Label.shape[0]}')
    intensities = []
    for l in df_sl_Label.Link:
        protein_path = Path(str(l).replace(".npy","_protein.png"))
        ori_fft = df.loc[df.index== l].values[0]
        intensity = plotting.get_protein_intensity(
            pro_path = protein_path, 
            shift_dict = shifts[l],
            ori_fft = ori_fft, 
            n_coef = n_coef, 
            inverse_func = inverse_func
            )
        
        intensities += [intensity.flatten()]
    tmp = np.nanmean(intensities, axis=0).reshape(intensity.shape)
    
    
    org_intensities = tmp/tmp.max()
    fig, ax = plt.subplots()
    for (xi,yi,intensities_layer) in zip(x_,y_,org_intensities):
        ax.scatter(xi, yi, c=intensities_layer, norm=norm)
    ax.axis("scaled")
    ax.set_facecolor('#541352FF')
    ax.axis("off")
    plt.savefig(os.path.join(d.cwd(), f"shapespace_plots/U2OS_{org}.png"), bbox_inches='tight')
    
    
#%%

COLORS = [
    '#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5',
    '#2196f3', '#03a9f4', '#00bcd4', '#009688', '#4caf50',
    '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800',
    '#ff5722', '#795548', '#9e9e9e', '#607d8b', '#dddddd',
    '#212121', '#ff9e80', '#ff6d00', '#ffff00', '#76ff03',
    '#00e676', '#64ffda', '#18ffff',
]

for org in list(all_locations.keys())[:-1]:
    img = imageio.imread(os.path.join(d.cwd(), f"shapespace_plots/U2OS_{org}.png"))
    img = interpolate_