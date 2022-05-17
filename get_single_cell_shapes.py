# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 08:23:22 2021

@author: trang.le

This code takes in images and cell masks and return single cell shapes
"""
import imaplib
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
import pickle 
from skimage.segmentation import clear_border

def bbox_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


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


def get_cell_nuclei_masks(image_id, cell_json, base_url):
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
    
def get_single_cell_mask(cell_mask, nuclei_mask, protein, keep_cell_list, save_path, rm_border=True, plot=True):
    if rm_border:
        nuclei_mask = clear_border(nuclei_mask)
        keep_value = np.unique(nuclei_mask)
        borderedcellmask = np.array([[x_ in keep_value for x_ in x] for x in cell_mask]).astype('uint8')
        cell_mask = cell_mask*borderedcellmask
    assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)) 
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


def get_cell_nuclei_masks2(encoded_image_id):
    cell_mask_ = imageio.imread(encoded_image_dir +'/' + encoded_image_id + '_mask.png')
    cell_regionprops = skimage.measure.regionprops(cell_mask_)

    nu = imageio.imread(encoded_image_dir +'/' + encoded_image_id + '_blue.png')
    mt = imageio.imread(encoded_image_dir +'/' + encoded_image_id + '_red.png')
    # Discard bordered cells because they have incomplete shapes
    nuclei_mask_0, _ = watershed_lab(nu, marker=None, rm_border=True)
    #nuclei_regionprops = skimage.measure.regionprops(nuclei_mask_0)
        
    marker = np.zeros_like(nuclei_mask_0)
    marker[nuclei_mask_0 > 0] = nuclei_mask_0[nuclei_mask_0 > 0]  # foreground
    cell_mask_0 = watershed_lab2(cell_mask_, marker=marker)
    cell_regionprops_0 = skimage.measure.regionprops(cell_mask_0)
    
    # Relabel
    nuclei_mask = np.zeros_like(cell_mask_)
    cell_mask = cell_mask_.copy()
    for region in cell_regionprops:
        new_label = region.label
        bbox_new = region.bbox
        old_label = None
        #if new_label == 8:
        #    breakme
        for r in cell_regionprops_0:
            bbox_old = r.bbox
            if bbox_iou(bbox_old, bbox_new) > 0.5:
                old_label = r.label
                print(new_label,old_label)
        if old_label == None:
            # If don't find any corresponding cell/nuclei, delete this mask
            print(f'Dont find cell {new_label}')
            cell_mask[cell_mask_ == new_label] = 0
        else:
            # If find something, update nuclei mask to the same index
            nuclei_mask[nuclei_mask_0 == old_label] = new_label
    assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask))
        
    protein = imageio.imread(encoded_image_dir +'/' + encoded_image_id + '_green.png')
    return cell_mask, nuclei_mask, protein

def get_single_cell_mask2(cell_mask, nuclei_mask, protein, keep_cell_list, save_path, plot=False):
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
def pilot_U2OS_kaggle2021test():
    base_dir = "C:/Users/trang.le/Desktop/annotation-tool"
    base_url = "https://if.proteinatlas.org"
    encoded_image_dir = f"{base_dir}/HPA-Challenge-2020-all/HPA_Kaggle_Challenge_2020/data_for_Kaggle/data"
    save_dir = "C:/Users/trang.le/Desktop/2D_shape_space/U2OS"
    # json_path = base_dir + "/HPA-Challenge-2020-all/segmentation/10093_1772_F9_7/annotation_all_ulrika.json"
    df = pd.read_csv(base_dir + "/final_labels_allversions.csv")
    df_test = pd.read_csv(base_dir + "/final_labels_allversions_current_withv6.csv")

    labels = pd.read_csv(f"{base_dir}/HPA-Challenge-2020-all/data_for_Kaggle/labels.csv")
    labels["Image_ID"] = [l.split("_")[0] for l in labels.ID] 
    labels["cell_ID"] = [str(int(l.split("_")[1]) - 1) for l in labels.ID] 
    mappings = pd.read_csv(f"{base_dir}/HPA-Challenge-2020-all/mappings.csv")
    labels = pd.merge(labels, mappings, on='Image_ID')
    labels["cell_id"] = labels["HPA_ID"] + '/' +  labels["cell_ID"]
    labels = pd.merge(labels, df_test, on = 'cell_id')

    df = labels[labels.atlas_name == "U-2 OS"]
    df = df[df.Label != 'Discard']
    imlist = list(set(df.image_id))
    error_list = []
    encoded_imlist = [mappings[mappings.HPA_ID==im].Image_ID.values[0] for im in imlist]

    finished_imlist = glob.glob(save_dir+'/*.npy')
    finished_imlist = [os.path.basename(t) for t in finished_imlist]
    finished_imlist = [t.rsplit('_',1)[0] for t in finished_imlist]
    finished_imlist = list(set(finished_imlist))

    for img_id, encoded_id in zip(imlist, encoded_imlist):
        if img_id in finished_imlist:
            continue
        df_img = df[df.image_id == img_id]
        cell_idx = [int(c.split('/')[1])+1 for c in df_img.cell_id]
        #json_path = max(
        #    glob.glob(
        #        f"{base_dir}/HPA-Challenge-2020-all/segmentation/{img_id}/annotation_*"
        #    ),
        #    key=os.path.getctime,
        #)
        try:
            #plot_complete_mask(json_path)
            #cell_mask, nuclei_mask, protein = get_cell_nuclei_masks(img_id, json_path, base_url)
            cell_mask, nuclei_mask, protein = get_cell_nuclei_masks2(encoded_id, encoded_image_dir)
            save_path = f"{save_dir}/{img_id}_"
            get_single_cell_mask2(cell_mask, nuclei_mask, protein, cell_idx, save_path)
        except:
            error_list += [img_id] 

def publicHPA():
    base_url = "/data/HPA-IF-images" #"https://if.proteinatlas.org"
    image_dir = "/data/kaggle-dataset/PUBLICHPA/images/test"
    mask_dir = "/data/kaggle-dataset/PUBLICHPA/mask/test"
    
    cell_line = "U-2 OS"
    save_dir = f"/data/2Dshapespace/{cell_line.replace(' ','_')}/cell_masks"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    log_dir = f"/data/2Dshapespace/{cell_line.replace(' ','_')}/logs"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # Load 
    finished_imlist = []
    if os.path.exists(f"{log_dir}/images_done.pkl"):
        with open(f"{log_dir}/images_done.pkl", "rb") as f:
            while True:
                try:
                    finished_imlist.append(pickle.load(f))
                except EOFError:
                    break        
            
    ifimages = pd.read_csv(f"{base_url}/IF-image.csv")
    ifimages = ifimages[ifimages.atlas_name==cell_line]
    ifimages["ID"] = [f.split("/")[-1][:-1] for f in ifimages.filename]
    im_df = pd.read_csv(f"{mask_dir}.csv")
    print(im_df.columns)
    imlist = list(set(im_df.ID.unique()).intersection(set(ifimages.ID)))
    success_list = open(f'{log_dir}/images_done.pkl', 'wb')
    error_list = open(f'{log_dir}/images_failed.pkl', 'wb')
    for img_id in imlist:
        if img_id in finished_imlist:
            continue
        df_img = im_df[im_df.ID == img_id]
        cell_idx = df_img.maskid.to_list()
        try:
            cell_mask = imageio.imread(f"{mask_dir}/{img_id}_cellmask.png")
            nuclei_mask = imageio.imread(f"{mask_dir}/{img_id}_nucleimask.png")
            protein = imageio.imread(f"{image_dir}/{img_id}_green.png")
            save_path = f"{save_dir}/{img_id}_"
            get_single_cell_mask(cell_mask, nuclei_mask, protein, cell_idx, save_path, rm_border=True, plot=False)
            pickle.dump(img_id, success_list)
        except:
            pickle.dump(img_id, error_list)
    success_list.close()
    error_list.close()

if __name__ == "__main__":   
    np.random.seed(42)  # for reproducibility
    #pilot_U2OS_kaggle2021test()
    publicHPA()
