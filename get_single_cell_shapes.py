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
import gzip
from descartes import PolygonPatch
from utils.helpers import (
    read_from_json,
    geojson_to_masks,
    watershed_lab,
    watershed_lab2,
)
import requests
from requests.auth import HTTPBasicAuth
import io
import glob

def plot_complete_mask(json_path):
    mask = read_from_json(json_path)
    img_size = (
        mask["bbox"][2] - mask["bbox"][0] + 1,
        mask["bbox"][3] - mask["bbox"][1] + 1,
    )
    img = np.zeros(img_size)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(img)
    for feature in mask["features"]:
        label = int(feature["properties"]["cell_idx"]) + 1
        coords = feature["geometry"]
        ax.add_patch(PolygonPatch(coords))
    ax.axis("off")
    plt.tight_layout()
    #fig.savefig("C:/Users/trang.le/Desktop/tmp.png", bbox_inches="tight")

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
    
def get_single_cell_mask(cell_mask, nuclei_mask, protein, keep_cell_list, save_path, plot=True):
    for region_c, region_n in zip(
        skimage.measure.regionprops(cell_mask), skimage.measure.regionprops(nuclei_mask)
    ):  
        if region_c.label not in keep_cell_list:
            continue
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
            plt.savefig(f"{save_path}{region_c.label}.jpg")

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


def get_cell_nuclei_masks2(encoded_image_id):
    cell_mask = imageio.imread(encoded_image_dir +'/' + encoded_image_id + '_mask.png')
    cell_regionprops = skimage.measure.regionprops(cell_mask)

    nu = imageio.imread(encoded_image_dir +'/' + encoded_image_id + '_blue.png')
    nuclei_mask_0, _ = watershed_lab(nu, marker=None, rm_border=False)
    #nuclei_regionprops = skimage.measure.regionprops(nuclei_mask_0)
        
    marker = np.zeros_like(nuclei_mask_0)
    marker[nuclei_mask_0 > 0] = nuclei_mask_0[nuclei_mask_0 > 0] + 1  # foreground
    cell_mask_0 = watershed_lab2(cell_mask, marker=marker)
    cell_regionprops_0 = skimage.measure.regionprops(cell_mask_0)
    
    # Relabel
    nuclei_mask = np.zeros_like(cell_mask)
    for region in cell_regionprops_0:
        old_label = region.label
        minr, minc, maxr, maxc = region.bbox
        print(region.bbox)
        for r in cell_regionprops:
            bbox = r.bbox
            print(bbox)
            #print(f'{bool(bbox[0]-1<=minr)} and {bool(bbox[1]-1 < minc)} and {bbox[2]+1 > maxr} and {bbox[0]+1 > maxc}')
            if (bbox[0]-1 <= minr) and (bbox[1]-1 <= minc) and (bbox[2]+1 >= maxr) and (bbox[3]+1 >= maxc):
                new_label = r.label
                print(old_label,new_label)
        nuclei_mask[nuclei_mask_0==old_label] = new_label
        print(np.unique(nuclei_mask))
    protein = imageio.imread(encoded_image_dir +'/' + encoded_image_id + '_green.png')
    return cell_mask, nuclei_mask, protein

def get_single_cell_mask2(cell_mask, nuclei_mask, protein, keep_cell_list, save_path, plot=True):
    regions_c = skimage.measure.regionprops(cell_mask)
    regions_n = skimage.measure.regionprops(nuclei_mask)
    for region_c in regions_c:  
        if region_c.label not in keep_cell_list:
            continue

        region_n = [region for region in regions_n if region.label==region_c.label][0]
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
            plt.savefig(f"{save_path}{region_c.label}.jpg")

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
#%% Test
import glob

np.random.seed(42)  # for reproducibility

base_dir = "C:/Users/trang.le/Desktop/annotation-tool"
base_url = "https://if.proteinatlas.org"
encoded_image_dir = 'C:/Users/trang.le/Desktop/annotation-tool/HPA-Challenge-2020-all/HPA_Kaggle_Challenge_2020/data_for_Kaggle/data'
save_dir = "C:/Users/trang.le/Desktop/2D_shape_space/U2OS"
# json_path = base_dir + "/HPA-Challenge-2020-all/segmentation/10093_1772_F9_7/annotation_all_ulrika.json"
df = pd.read_csv(base_dir + "/final_labels_allversions.csv")
df_test = pd.read_csv(base_dir + "/final_labels_allversions_current_withv6.csv")

labels = pd.read_csv('C:/Users/trang.le/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/labels.csv')
labels["Image_ID"] = [l.split("_")[0] for l in labels.ID] 
labels["cell_ID"] = [str(int(l.split("_")[1]) - 1) for l in labels.ID] 
mappings = pd.read_csv('C:/Users/trang.le/Desktop/annotation-tool/HPA-Challenge-2020-all/mappings.csv')
labels = pd.merge(labels, mappings, on='Image_ID')
labels["cell_id"] = labels["HPA_ID"] + '/' +  labels["cell_ID"]
labels = pd.merge(labels, df_test, on = 'cell_id')

df = labels[labels.atlas_name == "U-2 OS"]
df = df[df.Label != 'Discard']
imlist = list(set(df.image_id))
encoded_imlist = [mappings[mappings.HPA_ID==im].Image_ID.values[0] for im in imlist]
for img_id, encoded_id in zip(imlist, encoded_imlist):
    df_img = df[df.image_id == img_id]
    cell_idx = [int(c.split('/')[1]) for c in df_img.cell_id]
    json_path = max(
        glob.glob(
            f"{base_dir}/HPA-Challenge-2020-all/segmentation/{img_id}/annotation_*"
        ),
        key=os.path.getctime,
    )
    #plot_complete_mask(json_path)
    #cell_mask, nuclei_mask, protein = get_cell_nuclei_masks(img_id, json_path)
    cell_mask, nuclei_mask, protein = get_cell_nuclei_masks2(encoded_id)
    save_path = f"{save_dir}/{img_id}_"
    get_single_cell_mask2(cell_mask, nuclei_mask, protein, cell_idx, save_path)
