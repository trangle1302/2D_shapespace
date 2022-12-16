import os
import sys
sys.path.append("..") 
from imageio import imread, imwrite
import numpy as np
from utils import helpers
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.transform import resize
from warps import TPSpline, image_warp
import json
import pandas as pd
from tqdm import tqdm
import time


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

def avg_cell_landmarks(file_path):
    # Load average cell
    avg_cell = np.load(file_path)
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

    return pts_avg, (shape_x, shape_y) 

def main():   
    s = time.time()
    cell_line = 'U-2 OS'
    project_dir = f"/scratch/users/tle1302/2Dshapespace/{cell_line.replace(' ','_')}"
    shape_mode_path = f"{project_dir}/shapemode/{cell_line.replace(' ','_')}/fft_major_axis_polarized"  
    fft_dir = f"{project_dir}/fftcoefs/fft_major_axis_polarized"  
    data_dir = f"{project_dir}/cell_masks" 
    save_dir = f"{project_dir}/morphed_protein_avg" 
    plot_dir = f"{project_dir}/morphed_protein_avg_plots" 
    n_landmarks = 32 # number of landmark points for each ring, so final n_points to compute dx, dy will be 2*n_landmarks+1
    print(save_dir, plot_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Loading cell assignation into PC bins
    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json","r")
    cells_assigned = json.load(f)
    mappings = pd.read_csv("/scratch/users/tle1302/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border.csv")
    mappings = mappings[mappings.atlas_name=="U-2 OS"]
    mappings["cell_idx"] = [idx.split("_",1)[1] for idx in mappings.id]
    
    PC = "PC1"
    pc_cells = cells_assigned[PC]
    merged_bins = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
    org_percent = {}
    for i, bin_ in enumerate(merged_bins):
        ls = [pc_cells[b] for b in bin_]
        ls = helpers.flatten_list(ls)
        ls = [os.path.basename(l).replace(".npy","") for l in ls]
        df_sl = mappings[mappings.cell_idx.isin(ls)]
        df_sl = df_sl[df_sl.location.isin(LABEL_TO_ALIAS.values())] # rm Negative, Multi-loc
        org_percent[f"bin{i}"] = df_sl.target.value_counts().to_dict()
    
    df = pd.DataFrame(org_percent)
    print(df)

    #pts_avg, (shape_x, shape_y) = avg_cell_landmarks(f"{shape_mode_path}/Avg_cell.npz")
    
    with open(f"{fft_dir}/shift_error_meta_fft128.txt", "r") as F:
        lines = F.readlines()
    
    for i, bin_ in enumerate(merged_bins):
        ls = [pc_cells[b] for b in bin_]
        ls = helpers.flatten_list(ls)
        ls = [os.path.basename(l).replace(".npy","") for l in ls]
        df_sl = mappings[mappings.cell_idx.isin(ls)]
        df_sl = df_sl[df_sl.location.isin(LABEL_TO_ALIAS.values())] # rm Negative, Multi-loc
        #org_percent[f"bin{i}"] = df_sl.target.value_counts().to_dict()
        #print(df_sl.target.value_counts())
        print(f"processing {df_sl.shape[0]} cells")
        
        #images = []
        for org in ["Centrosome","IntermediateF","ActinF","NuclearM","NuclearB"]:            
            if not os.path.exists(f"{save_dir}/{PC}/{org}"):
                os.makedirs(f"{save_dir}/{PC}/{org}")
            ls_ = df_sl[df_sl.target == org].cell_idx.to_list()
            for img_id in tqdm(ls_, desc=f"{PC}_{org}"):
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
                
                pts_convex = (pts_avg + pts_ori) / 2
                warped1 = image_warp.warp_image(pts_ori, pts_convex, img_resized, plot=False, save_dir="")
                warped = image_warp.warp_image(pts_convex, pts_avg, warped1, plot=False, save_dir="")
                imwrite(f"{save_dir}/{PC}/{org}/{img_id}.png", (warped*255).astype(np.uint8))
                #images += [warped]
        

if __name__ == '__main__':
    main()