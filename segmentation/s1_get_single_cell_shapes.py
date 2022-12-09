# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 08:23:22 2021

@author: trangle1302

This code takes in images and cell masks and return single cell shapes
"""
import os
import skimage
import imageio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gzip
import sys
sys.path.append("..") 
from utils.helpers import (
    read_from_json,
    watershed_lab,
    watershed_lab2,
    rgb_2_gray_unique,
    bbox_iou
)

from utils.geojson_helpers import (
    geojson_to_masks,
    plot_complete_mask
)
import requests
from requests.auth import HTTPBasicAuth
import io
import glob
import pickle 
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

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
    
def get_single_cell_mask(cell_mask, nuclei_mask, protein, keep_cell_list, save_path, rm_border=True, remove_size=100, plot=False):
    if rm_border:
        nuclei_mask = skimage.segmentation.clear_border(nuclei_mask)
        keep_value = np.unique(nuclei_mask)
        borderedcellmask = np.array([[x_ in keep_value for x_ in x] for x in cell_mask]).astype('uint8')
        cell_mask = cell_mask*borderedcellmask
    assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)) 
    for region_c, region_n in zip(
        skimage.measure.regionprops(cell_mask), skimage.measure.regionprops(nuclei_mask)
    ):  
        if region_c.label not in keep_cell_list:
            continue
        if region_c.area < remove_size:
            continue
        # draw rectangle around segmented cell and
        # apply a binary mask to the selected region, to eliminate signals from surrounding cell
        minr, minc, maxr, maxc = region_c.bbox
        # get mask
        mask = cell_mask[minr:maxr, minc:maxc].astype(np.uint8)
        mask[mask != region_c.label] = 0
        mask[mask == region_c.label] = 1
        if True: #erose and dilate to remove the small line
            mask = skimage.morphology.erosion(mask, skimage.morphology.square(5))
            mask = skimage.morphology.dilation(mask, skimage.morphology.square(5))
            #mask = dilation(mask,square(3))

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


def get_cell_nuclei_masks2(encoded_image_id, encoded_image_dir):
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

def get_cell_nuclei_masks_ccd(parent_dir, img_id, cell_mask_extension = "w2cytooutline.png", nuclei_mask_extension = "w2nucleioutline.png"):
    #parent_dir = "/data/2Dshapespace/S-BIAD34/Files/HPA040393"
    cyto = imageio.imread(f"{parent_dir}/{img_id}_{cell_mask_extension}")
    cyto = rgb_2_gray_unique(cyto)
    nu = imageio.imread(f"{parent_dir}/{img_id}_{nuclei_mask_extension}")
    nu = rgb_2_gray_unique(nu)

    # Relabel the cytosol region based on nuclei labels
    cell_mask = np.zeros_like(cyto)
    nuclei_mask = np.zeros_like(nu) #nu.copy()
    nu_regionprops = skimage.measure.regionprops(nu)
    cyto_regionprops = skimage.measure.regionprops(cyto)
    matched_ID = []
    for region in nu_regionprops:
        new_label = region.label
        match = dict()
        x1 = [str(x) for x in region.coords]
        for r in cyto_regionprops:
            if r.label in matched_ID:
                continue
            x2 = [str(x) for x in r.coords]
            overlap_px = set(x1).intersection(x2)
            if len(overlap_px)> 0:
                match[str(r.label)] = len(overlap_px)
        if len(match.keys())==0:
            # If don't find any corresponding cell/nuclei, delete this mask
            print(f'Dont find cell {new_label}')
            #nuclei_mask[nu == new_label] = 0
        else:
            # if find multiple matches, pick the highest overlap. This case include the case with only 1 match
            highest_match= max(match, key= lambda x: match[x])
            #print(highest_match, new_label)
            cell_mask[cyto == int(highest_match)] = new_label
            nuclei_mask[nu == new_label] = new_label
            matched_ID += [int(highest_match)]
    assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask))
    
    cell_mask_ = skimage.morphology.erosion(nuclei_mask, skimage.morphology.square(3)) + cell_mask
    # remove small patches
    cell_mask_ = skimage.morphology.erosion(cell_mask_, skimage.morphology.square(5))
    cell_mask_ = skimage.morphology.dilation(cell_mask_, skimage.morphology.square(5))
    protein = imageio.imread(f"{parent_dir}/{img_id}_w4_Rescaled.tif")
    return cell_mask_, nuclei_mask, protein

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

def process_img(img_id, im_df, mask_dir, image_dir, save_dir, log_dir, cell_mask_extension = "cellmask.png", nuclei_mask_extension = "nucleimask.png"):
    df_img = im_df[im_df.ID == img_id]
    cell_idx = df_img.maskid.to_list()
    try:
        cell_mask = imageio.imread(f"{mask_dir}/{img_id}_{cell_mask_extension}")
        nuclei_mask = imageio.imread(f"{mask_dir}/{img_id}_{nuclei_mask_extension}")
        protein = imageio.imread(f"{image_dir}/{img_id}_green.png")
        save_path = f"{save_dir}/{img_id}_"
        get_single_cell_mask(cell_mask, nuclei_mask, protein, cell_idx, save_path, rm_border=True, plot=False)
        with open(f'{log_dir}/images_done.pkl', 'wb') as success_list:
            pickle.dump(img_id, success_list)
    except:
        with open(f'{log_dir}/images_failed.pkl', 'wb') as error_list:
            pickle.dump(img_id, error_list)

def process_img_ccd(ab_id, mask_dir, save_dir, log_dir, cell_mask_extension = "w2cytooutline.png", nuclei_mask_extension = "w2nucleioutline.png"):
    data_dir = os.path.join(mask_dir, ab_id)
    save_dir = os.path.join(save_dir, ab_id)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    img_ids = [os.path.basename(f).replace("_w3.TIF","") for f in glob.glob(f"{data_dir}/*_w3.TIF")]
    if len(img_ids) == 0:
        img_ids = [os.path.basename(f).replace("_w3.tif","") for f in glob.glob(f"{data_dir}/*_w3.tif")]
    #check if these images have nuclei mask, cytosol mask and protein channel
    img_ids_filtered = []
    for img_id in img_ids:
        if os.path.exists(f"{data_dir}/{img_id}_{cell_mask_extension}") and os.path.exists(f"{data_dir}/{img_id}_{nuclei_mask_extension}") and os.path.exists(f"{data_dir}/{img_id}_w4_Rescaled.tif"):
            img_ids_filtered += [img_id]
    #print(img_ids)
    for img_id in img_ids_filtered: #if True:#try:
        cell_mask, nuclei_mask, protein = get_cell_nuclei_masks_ccd(data_dir, img_id)
        imageio.imwrite(f"{save_dir}/cellmask.png", cell_mask)
        imageio.imwrite(f"{save_dir}/nuclei_mask.png", nuclei_mask)
        save_path = f"{save_dir}/{img_id}_"
        cell_idx = np.unique(cell_mask)
        cell_idx_finished = glob.glob(f"{save_path}*.npy")
        cell_idx_finished = [int(f.replace(save_path,"").replace(".npy","")) for f in cell_idx_finished]
        #print(cell_idx)
        #print(cell_idx_finished)
        cell_idx = list(set(cell_idx).difference(set(cell_idx_finished)))
        #print(cell_idx)
        if len(cell_idx) > 0:
            get_single_cell_mask(cell_mask, nuclei_mask, protein, cell_idx, save_path, rm_border=True, remove_size=20, plot=False)
        with open(f'{log_dir}/images_done.pkl', 'ab') as success_list:
            pickle.dump(img_id, success_list)
    if False:#except:
        with open(f'{log_dir}/images_failed.pkl', 'wb') as error_list:
            pickle.dump(img_id, error_list)

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
    print(f"{len(finished_imlist)} images done, processing the rest ...")
    num_cores = multiprocessing.cpu_count() -1 # save 1 core for some other processes
    ifimages = pd.read_csv(f"{base_url}/IF-image.csv")
    ifimages = ifimages[ifimages.atlas_name==cell_line]
    ifimages["ID"] = [f.split("/")[-1][:-1] for f in ifimages.filename]
    im_df = pd.read_csv(f"{mask_dir}.csv")
    print(im_df.columns)
    imlist = set(im_df.ID.unique()).intersection(set(ifimages.ID))
    print(f"...Found {len(imlist)} images with masks")
    imlist = list(imlist.difference(finished_imlist))
    print(f"...Processing {len(imlist)} ab x 5 img each with masks in {num_cores}")
    inputs = tqdm(imlist)
    processed_list = Parallel(n_jobs=num_cores)(delayed(process_img)(i, im_df, mask_dir, image_dir, save_dir, log_dir, cell_mask_extension = "cytooutline.png", nuclei_mask_extension = "cytooutline.png") for i in inputs)
    with open(f'{log_dir}/processedlist.pkl', 'wb') as f:
        pickle.dump(processed_list, f)


def cellcycle():
    base_url = "/data/2Dshapespace/S-BIAD34/Files"
    image_dir = "/data/2Dshapespace/S-BIAD34/Files"
    mask_dir = "/data/2Dshapespace/S-BIAD34/Files"
    
    cell_line = "U-2 OS"
    save_dir = "/data/2Dshapespace/S-BIAD34/cell_masks"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    log_dir = "/data/2Dshapespace/S-BIAD34/logs"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # Load 
    finished_imlist = []
    """
    if os.path.exists(f"{log_dir}/images_done.pkl"):
        with open(f"{log_dir}/images_done.pkl", "rb") as f:
            while True:
                try:
                    finished_imlist.append(pickle.load(f))
                except EOFError:
                    break        
    """
    print(f"{len(finished_imlist)} images done, processing the rest ...")
    num_cores = multiprocessing.cpu_count() - 14 # save 1 core for some other processes
    ifimages = pd.read_csv(f"{base_url}/experimentB-processed.txt", sep="\t")
    ablist = ifimages["Antibody id"].unique()
    print(f"...Found {len(ablist)} antibody folder with masks")
    print(f"...Processing {len(ablist)} images with masks in {num_cores}")
    #abid = "HPA040393"
    #process_img_ccd(abid, mask_dir, save_dir, log_dir)
    #done = pd.read_pickle(f'{log_dir}/images_done.pkl')
    #ablist = list(set(ablist).difference(set(done))))
    inputs = tqdm(ablist)
    import time
    s = time.time()
    processed_list = Parallel(n_jobs=num_cores)(delayed(process_img_ccd)(i, mask_dir, save_dir, log_dir) for i in inputs)
    with open(f'{log_dir}/processedlist.pkl', 'wb') as f:
        pickle.dump(processed_list, f)
    print(f"Finished in {(time.time() - s)/3600}h")

if __name__ == "__main__":   
    np.random.seed(42)  # for reproducibility
    #pilot_U2OS_kaggle2021test()
    #publicHPA()
    cellcycle()
