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
import pandas as pd
from scipy import ndimage as ndi
import gzip
from descartes import PolygonPatch
from utils.helpers import (
    read_from_json,
    geojson_to_masks,
    image_roll,
    shift_center_mass,
    resize_pad,
    watershed_lab,
    watershed_lab2,
)
import pywt
import requests
from requests.auth import HTTPBasicAuth
import io
from PIL import Image


def plot_complete_mask(json_path):
    mask = read_from_json(json_path)
    img_size = (
        mask["bbox"][2] - mask["bbox"][0] + 1,
        mask["bbox"][3] - mask["bbox"][1] + 1,
    )
    img = np.zeros(img_size)
    fig, ax = plt.subplot(1, 1, figsize=(20, 20))
    ax.imshow(img)
    for feature in mask["features"]:
        label = int(feature["properties"]["cell_idx"]) + 1
        coords = feature["geometry"]
        ax.add_patch(PolygonPatch(coords))
    ax.axis("off")
    plt.tight_layout()
    fig.savefig("C:/Users/trang.le/Desktop/tmp.png", bbox_inches="tight")

    # img = imageio.imread('C:/Users/trang.le/Desktop/tmp.png')
    # plt.imshow(img)


def get_cell_nuclei_masks(image_id, cell_json):
    mask_dict = geojson_to_masks(cell_json, mask_types=["labels"])
    cell_mask = mask_dict["labels"]

    ab, plate, well, sample = image_id.split("_")
    url = f"{base_url}/{plate}/{plate}_{well}_{sample}_blue.tif.gz"
    r = requests.get(url, auth=HTTPBasicAuth("trang", "H3dgeh0g1302"))
    f = io.BytesIO(r.content)
    tf = gzip.open(f).read()
    img = imageio.imread(tf, "tiff")
    nuclei_mask, _ = watershed_lab(img, marker=None, rm_border=True)

    url = f"{base_url}/{plate}/{plate}_{well}_{sample}_green.tif.gz"
    r = requests.get(url, auth=HTTPBasicAuth("trang", "H3dgeh0g1302"))
    f = io.BytesIO(r.content)
    tf = gzip.open(f).read()
    protein = imageio.imread(tf, "tiff")

    marker = np.zeros_like(nuclei_mask)
    marker[nuclei_mask > 0] = nuclei_mask[nuclei_mask > 0] + 1  # foreground
    cell_mask2 = watershed_lab2(cell_mask, marker=marker)

    return cell_mask2, nuclei_mask, protein


def get_single_cell_mask(cell_mask, nuclei_mask, protein, save_path, plot=False):
    for region_c, region_n in zip(
        skimage.measure.regionprops(cell_mask), skimage.measure.regionprops(nuclei_mask)
    ):
        # draw rectangle around segmented cell and
        # apply a binary mask to the selected region, to eliminate signals from surrounding cell
        minr, minc, maxr, maxc = region_c.bbox
        # get mask
        mask = cell_mask[minr:maxr, minc:maxc].astype(np.uint8)
        mask[mask != region_c.label] = 0
        mask[mask == region_c.label] = 1

        mask_n = nuclei_mask[minr:maxr, minc:maxc].astype(np.uint8)
        mask_n[mask_n != region_n.label] = 0
        mask_n[mask_n == region_n.label] = 1

        pr = protein[minr:maxr, minc:maxc].copy()
        pr[mask != 1] = 0

        if plot:
            plt.figure(figsize=(10, 10))
            plt.imshow(mask)
            plt.imshow(mask_n, alpha=0.5)

        if plot:
            fig = plt.imshow(pr)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False) 
            
        if plot:
            plt.figure(figsize=(10, 10))
            plt.imshow(mask)
            plt.imshow(mask_n, alpha=0.5)
            plt.axis("off")
            plt.tight_layout()
            # plt.savefig(f"{save_path}{region_c.label}.jpg")

        imageio.imwrite(f"{save_path}{region_c.label}_protein.png", pr)
        data = np.stack((mask, mask_n))
        np.save(f"{save_path}{region_c.label}.npy", data)
        data = np.dstack((mask, np.zeros_like(mask), mask_n)) * 255
        imageio.imwrite(f"{save_path}{region_c.label}.png", data)
        """
        data = np.expand_dims(data, axis=1)
        data = np.repeat(data, 10, axis=1)
        data = np.swapaxes(data, 2,3) #channel,z,h,w
        np.save(f'{save_path}{region_c.label}.npy', data)
        """


def wavelet(x, max_lev=3, label_levels=3):
    shape = x.shape
    # max_lev = 3       # how many levels of decomposition to draw
    # label_levels = 3  # how many levels to explicitly label on the plots

    fig, axes = plt.subplots(2, 4, figsize=[14, 8])
    for level in range(0, max_lev + 1):
        if level == 0:
            # show the original image before decomposition
            axes[0, 0].set_axis_off()
            axes[1, 0].imshow(x, cmap=plt.cm.gray)
            axes[1, 0].set_title("Image")
            axes[1, 0].set_axis_off()
            continue

        # plot subband boundaries of a standard DWT basis
        draw_2d_wp_basis(
            shape, wavedec2_keys(level), ax=axes[0, level], label_levels=label_levels
        )
        axes[0, level].set_title("{} level\ndecomposition".format(level))

        # compute the 2D DWT
        # c = pywt.wavedec2(x, 'db2', mode='periodization', level=level)
        c = pywt.wavedec2(x, "bior1.3", mode="periodization", level=level)

        # normalize each coefficient array independently for better visibility
        c[0] /= np.abs(c[0]).max()
        for detail_level in range(level):
            c[detail_level + 1] = [d / np.abs(d).max() for d in c[detail_level + 1]]
        # show the normalized coefficients
        arr, slices = pywt.coeffs_to_array(c)
        axes[1, level].imshow(arr, cmap=plt.cm.gray)
        axes[1, level].set_title("Coefficients\n({} level)".format(level))
        axes[1, level].set_axis_off()

    plt.tight_layout()
    plt.show()


#%% Test
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
from sklearn.decomposition import PCA
from aicsshparam import shtools, shparam
from skimage.morphology import ball, cube, octahedron
import glob

np.random.seed(42)  # for reproducibility

base_dir = "C:/Users/trang.le/Desktop/annotation-tool"
base_url = "https://if.proteinatlas.org"
save_dir = "C:/Users/trang.le/Desktop/2D_shape_space/U2OS_2"
# json_path = base_dir + "/HPA-Challenge-2020-all/segmentation/10093_1772_F9_7/annotation_all_ulrika.json"
df = pd.read_csv(base_dir + "/final_labels_allversions.csv")
df = df[df.atlas_name == "U-2 OS"]
imlist = list(set(df.image_id))
for img_id in imlist:
    json_path = max(
        glob.glob(
            f"{base_dir}/HPA-Challenge-2020-all/segmentation/{img_id}/annotation_*"
        ),
        key=os.path.getctime,
    )
    cell_mask, nuclei_mask, protein = get_cell_nuclei_masks(img_id, json_path)
    save_path = f"{save_dir}/{img_id}_"
    get_single_cell_mask(cell_mask, nuclei_mask, protein, save_path)

wavelet(mask_alligned)
#%%
import vtk
from vtk.util import numpy_support


def get_random_3d_shape():
    idx = np.random.choice([0, 1, 2], 1)[0]
    element = [ball, cube, octahedron][idx]
    label = ["ball", "cube", "octahedron"][idx]
    img = element(10 + int(10 * np.random.rand()))
    img = np.pad(img, ((1, 1), (1, 1), (1, 1)), mode="constant")
    img = img.reshape(1, *img.shape)
    # Rotate shapes to increase dataset variability.
    img = shtools.rotate_image_2d(image=img, angle=360 * np.random.rand()).squeeze()
    return label, img


def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax


def plot_3d(array_3d):
    ax = make_ax(True)
    ax.voxels(array_3d, edgecolors="gray")
    plt.show()


df_coeffs = pd.DataFrame([])
for i in range(5):
    # Get a random shape
    label, img = get_random_3d_shape()
    plot_3d(img)
    # Parameterize with L=4, which corresponds to50 coefficients
    # in total
    (coeffs, _), _ = shparam.get_shcoeffs(image=img, lmax=4)
    coeffs.update({"label": label})
    df_coeffs = df_coeffs.append(coeffs, ignore_index=True)

# img_2d = np.expand_dims(img[15,:,:], axis=0)
img_2d = np.expand_dims(mask, axis=0)
# (coeffs, _), _ = shparam.get_shcoeffs(image=img_2d, lmax=4)
(coeffs, grid_rec), (image_, mesh, grid, transform) = shparam.get_shcoeffs(
    image=img_2d, lmax=2
)
mse = shtools.get_reconstruction_error(grid, grid_rec)

coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
mesh.GetNumberOfCells()

img_2d = np.swapaxes(img_2d, 0, 2)  # VTK requires YXZ
imgdata = vtk.vtkImageData()
imgdata.SetDimensions(img.shape)
img_2d = img_2d.transpose(2, 1, 0)
img_output = img_2d.copy()
img_2d = img_2d.flatten()
arr = numpy_support.numpy_to_vtk(img_2d, array_type=vtk.VTK_FLOAT)
arr.SetName("Scalar")
imgdata.GetPointData().SetScalars(arr)
