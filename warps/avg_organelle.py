import os
import sys
sys.path.append("..") 
#from imageio import imread, imwrite # callisto
#from imageio.v2 import imread, imwrite # sherlock
import numpy as np
from utils import helpers
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, rotate
from skimage.transform import resize
from warps import image_warp
import json
import pandas as pd
from tqdm import tqdm
import time
import gc
import argparse

LABEL_TO_ALIAS = {
  0: 'Nucleoplasm',
  1: 'NuclearM',
  2: 'Nucleoli',
  3: 'NucleoliFC',
  4: 'NuclearS',
  5: 'NuclearB',
  6: 'EndoplasmicR',
  7: 'GolgiA',
  8: 'IntermediateF',
  9: 'ActinF',
  10: 'Microtubules',
  11: 'MitoticS',
  12: 'Centrosome',
  13: 'PlasmaM',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'VesiclesPCP',
  19: 'Negative',
  19:'Multi-Location',
}

all_locations = dict((v, k) for k,v in LABEL_TO_ALIAS.items())

def avg_cell_landmarks(ix_n, iy_n, ix_c, iy_c, n_landmarks=32):
    nu_centroid = helpers.find_centroid([(x_,y_) for x_, y_ in zip(ix_n, iy_n)])
    nu_centroid = [nu_centroid[0],nu_centroid[1]]
    print(f"Nucleus centroid of the avg shape: {nu_centroid}")
    
    # Move average shape from zero-centered coords to min=[0,0]
    min_x = np.min(ix_c)
    min_y = np.min(iy_c)
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
    #print(nu_contour.shape, cell_contour.shape)
    
    pts_avg = np.vstack([np.asarray(nu_centroid),
                        helpers.realign_contour_startpoint(nu_contour),
                        helpers.realign_contour_startpoint(cell_contour)])
    #print(pts_avg.max(), pts_avg.min(), cell_contour[:,0].max(), cell_contour[:,1].max())
    shape_x, shape_y = np.round(cell_contour[:,0].max()).astype('int'), np.round(cell_contour[:,1].max()).astype('int')

    return pts_avg, (shape_x, shape_y) 

def main():   
    s = time.time()
    cell_line = 'U-2 OS'
    alignment = "fft_cell_major_axis_polarized"
    parser = argparse.ArgumentParser()
    #parser.add_argument("--merged_bins", nargs='+',help="bin to investigate", type=int)
    parser.add_argument("--pc", help="principle component", type=str)
    parser.add_argument("--org", help="organelle class", type=str)
    args = parser.parse_args()
    org = args.org
    PC = args.pc
    print(f"Processing {org} in {PC}")
    try: #if True:
        from imageio import imread, imwrite # callisto
        project_dir = f"/data/2Dshapespace/{cell_line.replace(' ','_')}"
    except:
        from imageio.v2 import imread, imwrite # sherlock 
        project_dir = f"/scratch/users/tle1302/2Dshapespace/{cell_line.replace(' ','_')}"
    shape_mode_path = f"{project_dir}/shapemode/{alignment}_cell_nuclei_nux4"  
    fft_dir = f"{project_dir}/fftcoefs/{alignment}"
    data_dir = f"{project_dir}/cell_masks" 
    save_dir = f"{project_dir}/morphed_protein_avg_nux4" 
    plot_dir = f"{project_dir}/morphed_protein_avg_plots_nux4" 
    n_landmarks = 64 # number of landmark points for each ring, so final n_points to compute dx, dy will be 2*n_landmarks+1
    print(save_dir, plot_dir)
    os.makedirs(save_dir,exist_ok=True)
    os.makedirs(plot_dir,exist_ok=True)
    
    # Loading cell assignation into PC bins
    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json","r")
    cells_assigned = json.load(f)
    #mappings = pd.read_csv("/scratch/users/tle1302/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border.csv")
    mappings = pd.read_csv("/data/kaggle-dataset/publicHPA_umap/results/webapp/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border.csv")
    mappings = mappings[mappings.atlas_name=="U-2 OS"]
    mappings["cell_idx"] = [idx.split("_",1)[1] for idx in mappings.id]
    
    # created a folder where avg organelle for each bin is saved
    if not os.path.isdir(f"{save_dir}/{PC}"):
        os.makedirs(f"{save_dir}/{PC}")

    pc_cells = cells_assigned[PC]

    #merged_bins = [[0,1,2],[4,5,6],[8,9,10]]
    merged_bins = [[0],[1],[2],[3],[4],[5],[6]]

    org_percent = {}
    for i, bin_ in enumerate(merged_bins):
        ls = [pc_cells[b] for b in bin_]
        ls = helpers.flatten_list(ls)
        ls = [os.path.basename(l).replace(".npy","") for l in ls]
        df_sl = mappings[mappings.cell_idx.isin(ls)]
        df_sl = df_sl[df_sl.location.isin(LABEL_TO_ALIAS.values())] # rm Negative, Multi-loc
        org_percent[f"bin{i}"] = df_sl.target.value_counts().to_dict()
    
    df = pd.DataFrame(org_percent)
    #print(df)

    avg_cell_per_bin = np.load(f"{shape_mode_path}/shapevar_{PC}_cell_nuclei.npz")

    with open(f"{fft_dir}/shift_error_meta_fft128.txt", "r") as F:
        lines = F.readlines()
    
    for i, bin_ in enumerate(merged_bins):
        if len(bin_) == 1:
            n_coef = len(avg_cell_per_bin['nuc'][0])//2
            ix_n = avg_cell_per_bin['nuc'][bin_[0]][:n_coef]
            iy_n = avg_cell_per_bin['nuc'][bin_[0]][n_coef:]
            ix_c = avg_cell_per_bin['mem'][bin_[0]][:n_coef]
            iy_c = avg_cell_per_bin['mem'][bin_[0]][n_coef:]
            pts_avg, (shape_x, shape_y) = avg_cell_landmarks(ix_n, iy_n, ix_c, iy_c, n_landmarks = n_landmarks)  

        ls = [pc_cells[b] for b in bin_]
        ls = helpers.flatten_list(ls)
        ls = [os.path.basename(l).replace(".npy","") for l in ls]
        df_sl = mappings[mappings.cell_idx.isin(ls)]
        df_sl = df_sl[df_sl.location.isin(LABEL_TO_ALIAS.values())] # rm Negative, Multi-loc
        
        if not os.path.exists(f"{save_dir}/{PC}/{org}_bin{bin_[0]}.png"): #for org in ["Nucleoplasm","Nucleoli","NucleoliFC","EndoplasmicR","NuclearS","GolgiA","Microtubules","Mitochondria","VesiclesPCP","PlasmaM","Cytosol","NuclearS","ActinF","Centrosome","IntermediateF","NuclearM","NuclearB"]: 
            # 1 empty avg_img for each organelle_pc_bin combination 
            avg_img = np.zeros((shape_x+2, shape_y+2), dtype='float64')
            if not os.path.isdir(f"{plot_dir}/{PC}/{org}"):
                os.makedirs(f"{plot_dir}/{PC}/{org}")
            ls_ = df_sl[df_sl.target == org].cell_idx.to_list()
            #if os.path.exists(f"{save_dir}/{PC}/{org}_bin{bin_[0]}.png"):
            #    continue
            ls_ = [img_id for img_id in ls_ if os.path.exists(f"{data_dir}/{img_id}_protein.png")]
            ls_ = [img_id for img_id in ls_ if os.path.exists(f"{data_dir}/{img_id}.npy")]
            if len(ls_) > 20:
                import random
                ls_ = random.sample(ls_, 20)
            for img_id in tqdm(ls_, desc=f"{PC}_bin{bin_[0]}_{org}"):
                for line in lines:
                    if line.find(img_id) != -1 :
                        vals = line.strip().split(';')
                        break
                theta = float(vals[1])
                shift_c = (float(vals[2].split(',')[0].strip('(')),(float(vals[2].split(',')[1].strip(')'))))
                
                cell_shape = np.load(f"{data_dir}/{img_id}.npy")
                img = imread(f"{data_dir}/{img_id}_protein.png")
                if img.dtype == 'uint16':
                    img = (img / 256).astype(np.uint8)
                #print("Original image: ", img.max(), img.dtype, len(ls_))
                img = rotate(img, theta)
                nu_ = rotate(cell_shape[1,:,:], theta)
                cell_ = rotate(cell_shape[0,:,:], theta)

                img_resized = resize(img, (shape_x, shape_y), mode='constant')
                nu_resized = resize(nu_, (shape_x, shape_y), mode='constant') * 255
                cell_resized = resize(cell_, (shape_x, shape_y), mode='constant') * 255
                #print(f"rotated img max: {img.max()}, resized img max: {img_resized.max()}")
                #print(f"rotated nu max: {nu_.max()}, resized nu max: {nu_resized.max()}, rotated cell max: {cell_.max()}, resized cell max: {cell_resized.max()}")
                pts_ori = image_warp.find_landmarks(nu_resized, cell_resized, n_points=n_landmarks, border_points = False)
                
                pts_convex = (pts_avg + pts_ori) / 2
                warped1 = image_warp.warp_image(pts_ori, pts_convex, img_resized, plot=False, save_dir="")
                print(warped1.max(), img_resized.max())
                warped = image_warp.warp_image(pts_convex, pts_avg, warped1, plot=False, save_dir="")
                #imwrite(f"{save_dir}/{PC}/{org}/{img_id}.png", (warped*255).astype(np.uint8))

                # adding weighed contribution of this image
                print("Accumulated: ", avg_img.max(), avg_img.dtype, "Addition: ", warped.max(), warped.dtype,  (warped / len(ls_)).max())
                avg_img += warped / len(ls_)
                
                if np.random.choice([True,False], p=[0.001,0.999]):
                    # Plot landmark points at morphing
                    fig, ax = plt.subplots(1,5, figsize=(15,30))
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
                    fig.savefig(f"{plot_dir}/{PC}/{org}/{img_id}.png", bbox_inches='tight')
                    plt.close()
            print("Accumulated: ", avg_img.max(), avg_img.dtype, "Addition: ", warped.max(), warped.dtype)
            print(f"======>>> Saving to {save_dir}/{PC}/{org}_bin{bin_[0]}.png")
            imwrite(f"{save_dir}/{PC}/{org}_bin{bin_[0]}.png", (avg_img*255).astype(np.uint8))
            #:imwrite(f"{save_dir}/{PC}/bin{bin_[0]}_{org}.png", (avg_img*255).astype(np.uint8))
            gc.collect()
    print(f"Time elapsed: {(time.time() - s)/3600} h.")

if __name__ == '__main__':
    main()
