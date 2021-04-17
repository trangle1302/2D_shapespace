import os
import numpy as np
import pandas as pd
import glob
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import cmath

def fourier_coeffs(shape_coords, n=64):
    coords = shape_coords
    
    x = np.array([p[0] for p in coords])
    y = np.array([p[1] for p in coords])
    
    x_, y_ = equidistance(x,y, n_points=n)
    fft_x = np.fft.fft(x_)
    fft_y = np.fft.fft(y_)
    
    coeffs = [fft_x] + [fft_y]
    
    ix_ = np.fft.ifft(fft_x)
    iy_ = np.fft.ifft(fft_y)
    
    ix, iy = equidistance(ix_,iy_,len(coords))
    '''
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].plot(x,y)
    ax[0].axis('scaled')
    ax[1].plot(x, label = "x coord")
    ax[1].plot(y, label = "y coord")
    ax[1].legend()
    ax[2].plot(ix.real,iy.real)
    ax[2].axis('scaled')
    plt.tight_layout()
    '''
    error = (np.average(abs(x-ix)) + np.average(abs(y-iy)))/2
    
    return coeffs, error

def lowpassfilter(signal, thresh = 0.7, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

def wavelet_coefs(shape, wavelet_type='db5'):
    coords = find_contours(shape)
    
    x = np.array([p[0] for p in coords[0]])
    y = np.array([p[1] for p in coords[0]])
    
    cAx, cDx = pywt.dwt(x, wavelet_type)
    cAy, cDy = pywt.dwt(y, wavelet_type)
    
    #cAx = pywt.threshold(cAx, np.std(cAx)/2, mode='soft')
    #cAy = pywt.threshold(cAy, np.std(cAy)/2, mode='soft')

    ix = pywt.idwt(None, cDx, wavelet_type)
    iy = pywt.idwt(None, cDy, wavelet_type)
    
    coeffs = []
    '''
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    #ax[0].imshow(shape)
    ax[0].plot(x,y)
    ax[0].axis('scaled')
    ax[1].plot(x, label = "x coord")
    ax[1].plot(y, label = "y coord")
    ax[1].legend()
    ax[2].plot(ix,iy)
    ax[2].axis('scaled')
    plt.tight_layout()
    '''
    error = (np.average(abs(x-ix)) + np.average(abs(y-iy)))/2
    return coeffs, pos, error.real
    
def display_scree_plot(pca):
    '''Display a scree plot for the pca'''

    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red")
    
    for thres in [70,80]:
        idx = np.searchsorted(scree.cumsum(), thres)
        plt.plot(idx+1, scree.cumsum()[idx], c="red",marker='o')
        plt.annotate(f"{idx} PCs", xy=(idx+3, scree.cumsum()[idx]-5))
    plt.xlabel("Number of PCs")
    plt.ylabel("Percentage explained variance")
    plt.title("Scree plot")
    #plt.hlines(y=70, xmin = 0, xmax = len(scree), linestyles='dashed', alpha=0.5)
    #plt.vlines(x=np.argmax(scree.cumsum()>70), ymin = 0, ymax = 100, linestyles='dashed', alpha=0.5)
    #plt.hlines(y=80, xmin = 0, xmax = len(scree), linestyles='dashed', alpha=0.5)
    #plt.vlines(x=np.argmax(scree.cumsum()>80), ymin = 0, ymax = 100, linestyles='dashed', alpha=0.5)
    plt.show(block=False)
    
def calculate_feature_importance(pca, df_trans):
    df_dimred = {}
    loading = pca.components_.T * np.sqrt(pca.explained_variance_)
    for comp, pc_name in enumerate(df_trans.columns):
        load = loading[:, comp]
        pc = [v for v in load]
        apc = [v for v in np.abs(load)]
        total = np.sum(apc)
        cpc = [100 * v / total for v in apc]
        df_dimred[pc_name] = pc
        df_dimred[pc_name.replace("_PC", "_aPC")] = apc
        df_dimred[pc_name.replace("_PC", "_cPC")] = cpc
    df_dimred["features"] = df_trans.columns
    df_dimred = pd.DataFrame(df_dimred)
    df_dimred = df_dimred.set_index("features", drop=True)
    return df_dimred

def get_coefs_df(imlist, n_coef=32):
    fourier = pd.DataFrame()
    names = []
    error_n = []
    error_c = []
    for im in imlist:
        data = np.load(im)
        try:
            nuclei = data[1,:,:]   
            nuclei_coords = find_contours(nuclei)
            centroid = find_centroid(nuclei_coords[0])
            nuclei_coords = nuclei_coords[0] - centroid           
            fcoef_n, e_n = fourier_coeffs(nuclei_coords, n=n_coef) 
            
            cell = data[0,:,:]
            cell_coords = find_contours(cell)
            cell_coords = cell_coords[0] - centroid
            fcoef_c, e_c = fourier_coeffs(cell_coords, n=n_coef)
            
            error_c += [e_c]
            error_n += [e_n]
            fourier = fourier.append([np.concatenate([fcoef_c,fcoef_n]).ravel().tolist()], ignore_index=True)
            names += [im] 
        except:
            print("Oops!", sys.exc_info()[0], "occurred.")
            print(im)
            continue
    print(f'Get coefficients for {len(names)}/{len(imlist)} cells')
    print(f'Reconstruction error for nucleus: {np.average(error_n)}')    
    print(f'Reconstruction error for cell: {np.average(error_c)}')
    return fourier, names

import pathlib
d = pathlib.Path("C:/Users/trang.le/Desktop/2D_shape_space/U2OS")
imlist = [i for i in d.glob('*.npy')]
fourier_df = dict()
names_df = dict()
for n_coef in [32, 64]:
    df_, names_ = get_coefs_df(imlist, n_coef=n_coef)
    fourier_df[f'fft_{n_coef}'] = df_
    names_df[f'fft_{n_coef}'] = names_

#df_6 = fourier
#df_8 = fourier
#df_16 = fourier
#df_32 = fourier
# PCA
from sklearn.decomposition import PCA
n_coef=32
df = fourier_df['fft_32'].copy()
#sc = ComplexScaler()
#sc = ComplexNormalizer()
#sc.fit(df)
pca = ComplexPCA(n_components=df.shape[1])
pca.fit(df)#sc.transform(df))
display_scree_plot(pca)

matrix_of_features_transform = pca.transform(df)
pc_names = [f"PC{c}" for c in range(1, 1+len(pca.explained_variance_ratio_))]
pc_keep = [f"PC{c}" for c in range(1, 1+9)]
df_trans = pd.DataFrame(data=matrix_of_features_transform)
df_trans.columns=pc_names
df_trans[list(set(pc_names) - set(pc_keep))] = 0 #NOPE!!!
df_scaled_inv = pca.inverse_transform(df_trans)
#df_inv= sc.inverse_transform(df_scaled_inv)
df_inv = df_scaled_inv

for i, row in df_inv.iterrows():
    if i < 40:
        continue
    fcoef_c = row[0:n_coef*2]
    fcoef_n = row[n_coef*2:]
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(plt.imread(names_df['fft_32'][i].with_suffix('.jpg')))
    for fcoef in [fcoef_c, fcoef_n]:
        ix_ = np.fft.ifft(fcoef[0:n_coef])
        iy_ = np.fft.ifft(fcoef[n_coef:])
        ax[1].plot(ix_,iy_)
        ax[1].axis('scaled')
    if i > 200:
        breakme

def inverse_fft(fft_x,fft_y):
    ix = np.fft.ifft(fft_x)
    iy = np.fft.ifft(fft_y)
    return ix, iy

class PlotShapeModes():
    def __init__(self, pca, features_transform, n_coef):
        self.pca = pca
        self.matrix = features_transform
        self.n = n_coef
        self.midpoints = None
        self.std = None
        
        mean = self.matrix.mean(axis=0)
        self.midpoints = mean
        std = []
        for c in self.matrix:
            col = self.matrix[c]
            real = [x.real for x in col]
            imag = [x.imag for x in col]
            std +=  [complex(np.std(real),np.std(imag))]
        self.std = pd.Series(std, index = mean.index)
        self.equipoints = None
        self.get_equipoints()
    
    def plot_pc_dist(self, pc_name):
        cnums = self.matrix[pc_name]
        X = [x.real for x in cnums]
        Y = [x.imag for x in cnums]
        midpoint = self.midpoints[pc_name]
        '''
        p_std = []
        std_ = self.std[pc_name]
        for k in [-2,-1.5,-1,-0.5,0.5,1,1.5,2]:
            p_std += [midpoint + k*std_]
        '''    
        
        plt.scatter(X,Y, color='blue', alpha = 0.1)
        plt.scatter(midpoint.real,midpoint.imag,
                    label="star", marker="*", color='red')
        for p in self.equipoints[pc_name]:
            plt.scatter(p.real,p.imag,
                    label="star", marker="*", color='orange')
        plt.xlabel('real axis')
        plt.ylabel('imaginary axis')
        plt.title(pc_name)
        plt.show()
        
    def plot_avg_cell(self):
        fcoef = self.pca.inverse_transform(self.midpoints)
        fcoef_c = fcoef[0:n_coef*2]
        fcoef_n = fcoef[n_coef*2:]
        ix_n, iy_n = inverse_fft(fcoef_n[0:n_coef], fcoef_n[n_coef:])
        ix_c, iy_c = inverse_fft(fcoef_c[0:n_coef], fcoef_c[n_coef:])
        plt.title('Avg cell')
        plt.plot(ix_n,iy_n)
        plt.plot(ix_c,iy_c)
        plt.axis('scaled')

    def get_equipoints(self):
        points = dict()
        for c in self.matrix:
            col = self.matrix[c]
            real = [x.real for x in col]
            imag = [x.imag for x in col]
            r_, i_ = equidistance(real, imag, 9)
            points[c] = [complex(r,i) for r,i in zip(r_,i_)]
        self.equipoints = points
    
    def plot_shape_variation(self, pc_name):
        fig, ax = plt.subplots(1,9, figsize=(15,4))
        for i, p in enumerate(self.equipoints[pc_name]):        
            cell_coef = self.midpoints
            cell_coef[pc_name] = p 
            fcoef = self.pca.inverse_transform(cell_coef)
            fcoef_c = fcoef[0:n_coef*2]
            fcoef_n = fcoef[n_coef*2:]
            ix_n, iy_n = inverse_fft(fcoef_n[0:n_coef], fcoef_n[n_coef:])
            ix_c, iy_c = inverse_fft(fcoef_c[0:n_coef], fcoef_c[n_coef:])
            #ax[i].title(f'Cell at {}std')
            ax[i].plot(ix_n.real,iy_n.real)
            ax[i].plot(ix_c.real,iy_c.real)
            ax[i].axis('scaled')
        plt.show()
    
pm = PlotShapeModes(pca, matrix_of_features_transform, 64)
for pc in pc_keep:
    pm.plot_pc_dist(pc)
    pm.plot_shape_variation(pc)
    
    
## Allign the cells to the shortest distance from the nucleus to the cells down for eg
def assign_shape:
    