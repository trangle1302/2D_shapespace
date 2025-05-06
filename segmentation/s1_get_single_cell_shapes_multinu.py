# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 08:23:22 2021

@author: trangle1302

This code takes in images and cell masks and return single cell shapes
"""

import os
import skimage
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gzip
import time
import sys
sys.path.append("..")
from utils.helpers import (
    read_from_json,
    watershed_lab,
    watershed_lab2,
    rgb_2_gray_unique,
    bbox_iou,
)
import io
import glob
import pickle
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
from shapely.geometry import Polygon
import cv2

B2AI_channel_map = {
    "C0.tif" : "DAPI",
    "C1.tif" : "ER",
    "C2.tif" : "BF",
    "C3.tif" : "Protein",
    "C4.tif" : "Microtubules"
}
def coords_to_str(coords):
    l = []
    for c in coords:
        l += [",".join(str(x) for x in c)]
    return l

def get_single_cell_mask2(
    cell_mask, nuclei_mask, protein, keep_cell_list, save_path, plot=False
):
    regions_c = skimage.measure.regionprops(cell_mask)
    regions_n = skimage.measure.regionprops(nuclei_mask)
    for region_c in regions_c:
        if region_c.label not in keep_cell_list:
            continue

        region_n = [region for region in regions_n if region.label == region_c.label][0]
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
            plt.savefig(f"{save_path}{region_c.label}.jpg", bbox_inches="tight")

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
        
def sharpen(image):
    image = skimage.img_as_float(image)
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = skimage.exposure.rescale_intensity(image, in_range=(p2, p98))
    return img_rescale
    
def draw_polygon_boundaries0(image, labeled_mask, color_brg=[0, 0, 255]):
    """
    Draw the boundary of every labeled region (polygon) in red on the given image.

    :param image: A 3-channel image (W, H, C)
    :param mask: A 2D array with labeled regions (each unique integer corresponds to a region)
    :return: The image with polygon boundaries drawn on it
    """
    # Extract properties of each region
    '''
    regions = skimage.measure.regionprops(labeled_mask)
    for region in regions:
        # Get the coordinates of the polygon perimeter
        #perimeter_coords = region.coords
        contours = skimage.measure.find_contours(region.coords)
        # Draw the perimeter in red
    '''
    contours = skimage.measure.find_contours(labeled_mask,1)
    for coords in contours:
        for coord in np.round(coords).astype('int'):
            print(coord)
            rr, cc = coord
            image[rr, cc] = color_brg  # [0,0,255] = red, [255,0,0] = blue
    return image
    
def draw_polygon_boundaries(image, labeled_mask, color_brg=(0, 0, 255)):
    """
    Draw the boundary of every labeled region (polygon) in red on the given image using OpenCV.

    :param image: A 3-channel image (W, H, C)
    :param mask: A 2D array with labeled regions (each unique integer corresponds to a region)
    :return: The image with polygon boundaries drawn on it
    """
    regions = skimage.measure.regionprops(labeled_mask)

    for region in regions:
        # Create a binary mask for the current region
        region_mask = (labeled_mask == region.label).astype(np.uint8)
        # Find contours in the binary mask
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw the contours in red on the image
        cv2.drawContours(image, contours, -1, color_brg, thickness=2)  # Red color in BGR
    return image
    
def get_valid_poly(regions, min_area):
    polygons = {}
    for region in regions:
        if region.area > min_area:
            poly = Polygon(region.coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid:
                polygons[region.label] = poly
                #polygons.append(poly)
    return polygons
    
def find_polygon_matches(mask1, mask2, min_area=100):
    """
    Find matches between polygons in two masks (all nuclei and all cells).
    
    :param mask1: 2D numpy array where each region is labeled with a unique integer.
    :param mask2: 2D numpy array where each region is labeled with a unique integer.
    :return: List of tuples where each tuple contains two elements: index of region in mask1 and index of region in mask2.
    """
    regions1 = skimage.measure.regionprops(mask1)
    regions2 = skimage.measure.regionprops(mask2)
    
    polygons1 = get_valid_poly(regions1, min_area)
    polygons2 = get_valid_poly(regions2, min_area)
    matches = {}
    for i, poly1 in polygons1.items():
        matches[i] = []
        for j, poly2 in polygons2.items():
            if poly1.intersects(poly2):
                matches[i].append(j)
    return matches
    
def process_image_b2ai(cell_mask_path, save_path, plot=False):
    names = cell_mask_path.split('/')
    save_path = f"{save_path}/{names[-4]}/{names[-3]}/"
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    cell_mask = imageio.imread(cell_mask_path)
    nuclei_mask = imageio.imread(cell_mask_path.replace("cytomask.png","nucleimask.png"))
    protein_path = cell_mask_path.replace("cytomask.png","C3.tif")
    protein = imageio.imread(protein_path)
    
    nuclei_mask = skimage.segmentation.clear_border(nuclei_mask)
    
    matches = find_polygon_matches(cell_mask, nuclei_mask)
    #print(matches)
    regions_c = skimage.measure.regionprops(cell_mask)
    regions_n = skimage.measure.regionprops(nuclei_mask)
    for cell, nu in matches.items():
        if len(nu) == 0 or len(nu) > 1:
            if plot:
                #print("multinuclei or no nuclei match")
                cell_mask[cell_mask == cell] = 0
                if len(nu) > 0:
                    for nu_ in nu:
                        nuclei_mask[nuclei_mask == nu_] = 0
            continue

        region_c = [region for region in regions_c if region.label == cell][0]
        region_n = [region for region in regions_n if region.label == nu[0]][0]
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
        
        imageio.imwrite(f"{save_path}{region_c.label}_protein.png", pr)
        data = np.stack((mask, mask_n))
        np.save(f"{save_path}{region_c.label}.npy", data)
        data = np.dstack((mask, np.zeros_like(mask), mask_n)) * 255
        imageio.imwrite(f"{save_path}{region_c.label}.png", data)
    
    if plot:
        image = np.dstack([sharpen(imageio.imread(protein_path.replace('C3','C0'))),
                        sharpen(protein),
                        sharpen(imageio.imread(protein_path.replace('C3','C4')))])
        #print(image.shape, image.max(), image.min(), image.dtype)
        image = (image*255).astype('uint8')
        image = draw_polygon_boundaries(image, cell_mask, color_brg=(0, 0, 255))
        image = draw_polygon_boundaries(image, nuclei_mask, color_brg=(255, 0, 255))
        imageio.imwrite(f"{save_path}_QC.png", image)
        
def perturbation_B2AI():
    base_url = "/scratch/users/tle1302/2Dshapespace/B2AI/MDA-MB-468/Tiffs/B2AI-2023-1/"
    save_dir = "/scratch/users/tle1302/2Dshapespace/B2AI/cell_masks"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    imlist = glob.glob(f'{base_url}/*/*untreated*/z01/cytomask.png')
    #imlist = [f"{base_dir}/CAB080425_KAT2A/B2AI_1_untreated_E2_R2/z01/cytomask.png",
    #        f"{base_dir}/CAB080428_HDAC1/B2AI_1_Paclitaxel_B3_R4/z01/cytomask.png",
    #        f"{base_dir}/CAB080429_BRPF1/B2AI_1_untreated_D10_R7/z01/cytomask.png",
    #        ]
    imlist.sort()
    inputs = tqdm(imlist)
    s = time.time()
    num_cores = multiprocessing.cpu_count() -1
    print(f'Processing in {num_cores} cores')
    processed_list = Parallel(n_jobs=num_cores)(
        delayed(process_image_b2ai)(i, save_dir, plot=True) for i in inputs
    )
    with open(f"{log_dir}/processedlist.pkl", "wb") as f:
        pickle.dump(processed_list, f)
    print(f"Finished in {(time.time() - s)/3600}h")
    
    sc_stats_save_path = f"{cfg.PROJECT_DIR}/single_cell_statistics.csv"
    
if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility
    import configs.config as cfg

    # pilot_U2OS_kaggle2021test()
    # publicHPA(cell_line=cfg.CELL_LINE)
    # cellcycle()
    perturbation_B2AI()
    
    """ Test
    cell_mask_path = '/scratch/users/tle1302/2Dshapespace/B2AI/MDA-MB-468/Tiffs/B2AI-2023-1/CAB080425_KAT2A/B2AI_1_untreated_E2_R2/z01/cytomask.png'
    nuclei_mask_path = '/scratch/users/tle1302/2Dshapespace/B2AI/MDA-MB-468/Tiffs/B2AI-2023-1/CAB080425_KAT2A/B2AI_1_untreated_E2_R2/z01/nucleimask.png'
    protein_path = '/scratch/users/tle1302/2Dshapespace/B2AI/MDA-MB-468/Tiffs/B2AI-2023-1/CAB080425_KAT2A/B2AI_1_untreated_E2_R2/z01/C3.tif'
    save_path = "/scratch/users/tle1302/2Dshapespace/B2AI/cell_masks/CAB080425_KAT2A/B2AI_1_untreated_E2_R2/"
    os.makedirs(save_path, exist_ok=True)
    process_image_b2ai(cell_mask_path, save_path, plot=True)
    """