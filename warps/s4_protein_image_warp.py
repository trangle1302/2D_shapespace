import os
import sys
sys.path.append("..") 
from imageio import imread, imwrite
import numpy as np
from utils import helpers
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
from scipy.ndimage import rotate
from skimage.transform import resize
from warps import TPSpline, image_warp
import json
import pandas as pd
from tqdm import tqdm
import time

def main():   
    s = time.time()
    cell_line = 'U-2 OS'
    project_dir = f"/data/2Dshapespace/{cell_line.replace(' ','_')}"
    shape_mode_path = f"{project_dir}/shapemode/{cell_line.replace(' ','_')}/fft_major_axis_polarized"  
    fft_dir = f"{project_dir}/fftcoefs/fft_major_axis_polarized"  
    data_dir = f"{project_dir}/cell_masks" 
    save_dir = f"{project_dir}/morphed_protein_avg" 
    plot_dir = f"{project_dir}/morphed_protein_avg_plots" 
    n_landmarks = 32 # number of landmark points for each ring, so final n_points to compute dx, dy will be 2*n_landmarks+1

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Load average cell
    avg_cell = np.load(f"{shape_mode_path}/Avg_cell.npz")
    nu_centroid = [0,0]
    ix_n = avg_cell['ix_n']
    iy_n = avg_cell['iy_n']
    ix_c = avg_cell['ix_c']
    iy_c = avg_cell['iy_c']
    
    # Move average shape from zero-centered coords to min=[0,0]
    min_x = np.min(ix_c)
    min_y = np.min(iy_c)
    print(min_x,min_y)
    nu_centroid[0] -= min_x
    nu_centroid[1] -= min_y
    ix_n -= min_x
    iy_n -= min_y
    ix_c -= min_x
    iy_c -= min_y

    if len(ix_n) != n_landmarks:
        ix_n, iy_n = helpers.equidistance(ix_n, iy_n, n_points=n_landmarks)
        ix_c, iy_c = helpers.equidistance(ix_c, iy_c, n_points=n_landmarks)
    nu_contour = np.stack([ix_n, iy_n]).T
    cell_contour = np.stack([ix_c, iy_c]).T
    print(nu_contour.shape, cell_contour.shape)
    
    pts_avg = np.vstack([np.asarray(nu_centroid),
                        helpers.realign_contour_startpoint(nu_contour),
                        helpers.realign_contour_startpoint(cell_contour)])
    print(pts_avg.max(), pts_avg.min(), cell_contour[:,0].max(), cell_contour[:,1].max())
    shape_x, shape_y = np.round(cell_contour[:,0].max()).astype('int'), np.round(cell_contour[:,1].max()).astype('int')

    # Loading cell assignation into PC bins
    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json","r")
    cells_assigned = json.load(f)
    mappings = pd.read_csv("/data/kaggle-dataset/publicHPA_umap/results/webapp/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border.csv")
    mappings = mappings[mappings.atlas_name=="U-2 OS"]
    mappings["cell_idx"] = [idx.split("_",1)[1] for idx in mappings.id]
    
    pc_cells = cells_assigned['PC1']
    merged_bins = [[0,1,2,3,4],[5],[6,7,8,9,10]]
    #imlist = pc_cells[5]
    #print(imlist[:3])

    with open(f"{fft_dir}/shift_error_meta_fft128.txt", "r") as F:
        lines = F.readlines()
    for i, bin_ in enumerate(merged_bins):
        if i!=1:
            continue
        ls = [pc_cells[b] for b in bin_]
        ls = helpers.flatten_list(ls)
        ls = [os.path.basename(l).replace(".npy","") for l in ls]
        # df_sl = mappings[mappings.cell_idx.isin(ls)]
        # print(df_sl.target.value_counts())
        print(f"processing {len(ls)} cells")

        for img_id in tqdm(ls):
            for line in lines:
                if line.find(img_id) != -1 :
                    vals = line.strip().split(',')
                    break
            theta = np.float(vals[1])
            shift_c = (np.float(vals[2].strip('(')),(np.float(vals[3].strip(')'))))
            
            cell_shape = np.load(f"{data_dir}/{img_id}.npy")
            img = imread(f"{data_dir}/{img_id}_protein.png")
            #print(np.unique(cell_shape), img.dtype, img.max())
            
            img = rotate(img, theta)
            nu_ = rotate(cell_shape[1,:,:], theta)
            cell_ = rotate(cell_shape[0,:,:], theta)
            img_resized = resize(img, (shape_x, shape_y), mode='constant')
            nu_resized = resize(nu_, (shape_x, shape_y), mode='constant') * 255
            cell_resized = resize(cell_, (shape_x, shape_y), mode='constant') * 255
            #print(f"rotated img max: {img.max()}, resized img max: {img_resized.max()}")
            #print(f"rotated nu max: {nu_.max()}, resized nu max: {nu_resized.max()}, rotated cell max: {cell_.max()}, resized cell max: {cell_resized.max()}")
            pts_ori = image_warp.find_landmarks(nu_resized, cell_resized, n_points=32, border_points = False)
            
            """ TODO: fix convex hull
            convex_hull_nu = convex_hull_image(nu_resized)
            convex_hull_cell = convex_hull_image(cell_resized)
            pts_convex = image_warp.find_landmarks(convex_hull_nu, convex_hull_cell, n_points=32, border_points = False)
            """
            pts_convex = (pts_avg + pts_ori) / 2
            # TODO: how to preserve spotty pattern?
            warped1 = image_warp.warp_image(pts_ori, pts_convex, img_resized, plot=False, save_dir="")
            warped = image_warp.warp_image(pts_convex, pts_avg, warped1, plot=False, save_dir="")
            imwrite(f"{save_dir}/{img_id}.png", (warped*255).astype(np.uint8))
            #print(warped.max(), warped.dtype)
            fig, ax = plt.subplots(1,5, figsize=(15,30), sharex=True, sharey=True)
            ax[0].imshow(nu_, alpha = 0.3)
            ax[0].imshow(cell_, alpha = 0.3)
            ax[0].set_title('original shape')            
            ax[1].imshow(nu_resized, alpha = 0.3)
            ax[1].imshow(cell_resized, alpha = 0.3)
            ax[1].set_title('resized shape+protein')
            ax[2].imshow(img_resized)
            ax[2].scatter(pts_ori[:,1], pts_ori[:,0], c=np.arange(len(pts_ori)),cmap='Reds')
            ax[2].set_title('resized protein channel')
            ax[3].imshow(warped1)
            ax[3].scatter(pts_convex[:,1], pts_convex[:,0], c=np.arange(len(pts_ori)),cmap='Reds')
            ax[3].set_title('ori_shape to midpoint')
            ax[4].imshow(warped)
            ax[4].scatter(pts_avg[:,1], pts_avg[:,0], c=np.arange(len(pts_ori)),cmap='Reds')
            ax[4].set_title('midpoint to avg_shape')
            fig.savefig(f"{plot_dir}/{img_id}.png", bbox_inches='tight')
    print(f"Finished in {(time.time() - s)/60} min")

if __name__ == '__main__':
    main()