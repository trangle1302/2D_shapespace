# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 08:23:22 2021

@author: trang.le

This code takes in images and cell masks and return single cell shapes
"""
import os
import skimage
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy import ndimage as ndi
import gzip
from descartes import PolygonPatch
from utils import read_from_json, geojson_to_masks, image_roll, shift_center_mass, resize_pad
from skimage.segmentation import clear_border
import pywt

def plot_complete_mask(json_path):
    mask = read_from_json(json_path)
    img_size = (mask['bbox'][2]-mask['bbox'][0]+1,mask['bbox'][3]-mask['bbox'][1]+1)
    img = np.zeros(img_size)
    fig, ax = plt.subplot(1,1, figsize=(20,20))
    ax.imshow(img)
    for feature in mask['features']:
        label = int(feature['properties']['cell_idx']) +1
        coords = feature['geometry']
        ax.add_patch(PolygonPatch(coords))
    ax.axis('off')
    plt.tight_layout()
    fig.savefig('C:/Users/trang.le/Desktop/tmp.png',bbox_inches='tight')
    
    #img = imageio.imread('C:/Users/trang.le/Desktop/tmp.png')
    #plt.imshow(img)
    
def get_single_cell_mask(json_path):    
    mask_dict = geojson_to_masks(json_path, mask_types=["labels"]) 
    labels = mask_dict['labels']
    for region in skimage.measure.regionprops(labels):
        region_label = region.label
        # draw rectangle around segmented cell and
        # apply a binary mask to the selected region, to eliminate signals from surrounding cell
        minr, minc, maxr, maxc = region.bbox
        # get mask
        mask = labels[minr:maxr,minc:maxc].astype(np.uint8)
        mask[mask != region.label] = 0
        mask[mask == region.label] = 1
        mask_resized = resize_pad(mask, ratio=0.5, size=256)
        centroid = np.round(skimage.measure.centroid(mask_resized))
        
        # allign the centroid
        mask_centered = image_roll(mask_resized, centroid)
        # center to the center of mass of the nucleus
        # fig = shift_center_mass(mask)
        
        # align cell to the 1st major axis  
        theta=region.orientation*180/np.pi #radiant to degree conversion
        mask_alligned = ndi.rotate(mask_centered, 90-theta)
            
        name = "%s_cell%s.%s" % (json_path.split('/')[-2],region_label, "png")
        name = name.replace("/", "_")
        
        fig, ax = plt.subplots(1,4,figsize=(10,30))
        ax[0].imshow(mask)
        ax[1].imshow(mask_resized)
        ax[2].imshow(mask_centered)
        ax[3].imshow(mask_alligned)
        plt.show()
        
        # Haar wavelet transformation, cD=Approximation coef, cA=Detail coef
        cA, cD = pywt.dwt(mask_alligned, 'haar')
        # reconstruction signal
        y = pywt.idwt(cA, cD, 'haar')
        
        coeffs2 = pywt.dwt2(mask_alligned, 'bior1.3')
        # Approximation, (Horizonal details, Vertical details, Diagonal details)
        LL, (LH, HL, HH) = coeffs2
        #savepath = os.path.join(args.imgOutput, name)
        #skimage.io.imsave(savepath, fig)
        
        np.fft.fft(mask_alligned, n=None, axis=-1, norm=None)
        breakme

def wavelet(x, max_lev = 3, label_levels=3):
    shape = x.shape
    #max_lev = 3       # how many levels of decomposition to draw
    #label_levels = 3  # how many levels to explicitly label on the plots
    
    fig, axes = plt.subplots(2, 4, figsize=[14, 8])
    for level in range(0, max_lev + 1):
        if level == 0:
            # show the original image before decomposition
            axes[0, 0].set_axis_off()
            axes[1, 0].imshow(x, cmap=plt.cm.gray)
            axes[1, 0].set_title('Image')
            axes[1, 0].set_axis_off()
            continue
    
        # plot subband boundaries of a standard DWT basis
        draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level],
                         label_levels=label_levels)
        axes[0, level].set_title('{} level\ndecomposition'.format(level))
    
        # compute the 2D DWT
        #c = pywt.wavedec2(x, 'db2', mode='periodization', level=level)
        c = pywt.wavedec2(x, 'bior1.3', mode='periodization', level=level)
        
        # normalize each coefficient array independently for better visibility
        c[0] /= np.abs(c[0]).max()
        for detail_level in range(level):
            c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
        # show the normalized coefficients
        arr, slices = pywt.coeffs_to_array(c)
        axes[1, level].imshow(arr, cmap=plt.cm.gray)
        axes[1, level].set_title('Coefficients\n({} level)'.format(level))
        axes[1, level].set_axis_off()
    
    plt.tight_layout()
    plt.show()
#%% Test
import numpy as np
import pywt
from matplotlib import pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis     
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from aicsshparam import shtools, shparam
from skimage.morphology import ball, cube, octahedron
np.random.seed(42) # for reproducibility

json_path = "C:/Users/trang.le/Desktop/annotation-tool/HPA-Challenge-2020-all/segmentation/10093_1772_F9_7/annotation_all_ulrika.json"



wavelet(mask_alligned)
#%%
import vtk
from vtk.util import numpy_support
def get_random_3d_shape():
    idx = np.random.choice([0, 1, 2], 1)[0]
    element = [ball, cube, octahedron][idx]
    label = ['ball', 'cube', 'octahedron'][idx]
    img = element(10 + int(10 * np.random.rand()))
    img = np.pad(img, ((1, 1), (1, 1), (1, 1)), mode='constant')
    img = img.reshape(1, *img.shape)
    # Rotate shapes to increase dataset variability.
    img = shtools.rotate_image_2d(
        image=img,
        angle=360 * np.random.rand()
    ).squeeze()
    return label, img


def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax

def plot_3d(array_3d):
    ax = make_ax(True)
    ax.voxels(array_3d, edgecolors='gray')
    plt.show()


df_coeffs = pd.DataFrame([])
for i in range(5):
    # Get a random shape
    label, img = get_random_3d_shape()
    plot_3d(img)
    # Parameterize with L=4, which corresponds to50 coefficients
    # in total
    (coeffs, _), _ = shparam.get_shcoeffs(image=img, lmax=4)
    coeffs.update({'label': label})
    df_coeffs = df_coeffs.append(coeffs, ignore_index=True)

#img_2d = np.expand_dims(img[15,:,:], axis=0)
img_2d = np.expand_dims(mask, axis=0)
#(coeffs, _), _ = shparam.get_shcoeffs(image=img_2d, lmax=4)
(coeffs, grid_rec), (image_, mesh, grid, transform) = shparam.get_shcoeffs(image=img_2d, lmax=2)
mse = shtools.get_reconstruction_error(grid, grid_rec)

coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
mesh.GetNumberOfCells()

img_2d = np.swapaxes(img_2d, 0, 2) #VTK requires YXZ
imgdata = vtk.vtkImageData()
imgdata.SetDimensions(img.shape)
img_2d = img_2d.transpose(2, 1, 0)
img_output = img_2d.copy()
img_2d = img_2d.flatten()
arr = numpy_support.numpy_to_vtk(img_2d, array_type=vtk.VTK_FLOAT)
arr.SetName("Scalar")
imgdata.GetPointData().SetScalars(arr)
