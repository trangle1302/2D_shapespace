import os
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.filters import threshold_minimum, threshold_otsu
from skimage.metrics import structural_similarity
from scipy.stats import pearsonr
import json
from utils import helpers
import argparse
from imageio import imread, imwrite
import glob

def correlation(value_dict, method_func, mask):
    cor_mat = np.zeros((len(value_dict), len(value_dict)))
    for i, (k1, v1) in enumerate(value_dict.items()):
        for j, (k2, v2) in enumerate(value_dict.items()):
            if True:
                v1_ = (np.zeros_like(mask) if v1.max()==0 else v1.flatten()[mask])
                v2_ = (np.zeros_like(mask) if v2.max()==0 else v2.flatten()[mask])
                try:
                    cor_mat[i, j] = method_func(v1, v2)
                except:
                    (cor_mat[i, j],_) = method_func(v1_, v2_)
            else:
                try:
                    cor_mat[i, j] = method_func(v1, v2)
                except:
                    (cor_mat[i, j],_) = method_func(v1.flatten(), v2.flatten())
    return cor_mat

def unmerge_label(
    mappings_df,
    merged_label="VesiclesPCP",
    subcomponents=[
        "Lipid droplets",
        "Endosomes",
        "Lysosomes",
        "Peroxisomes",
        "Vesicles",
        "Cytoplasmic bodies",
    ],
):
    mappings_df["sc_locations"] = ""
    mappings_df["sc_target"] = ""
    for i, r in mappings_df.iterrows():
        if r.target == merged_label:
            sc_l = [l for l in r.locations.split(",") if l in subcomponents]
            mappings_df.loc[i, "sc_locations"] = ",".join(sc_l)
            if len(sc_l) > 1:
                mappings_df.loc[i, "sc_target"] = "Multi-Location"
            else:
                mappings_df.loc[i, "sc_target"] = sc_l[0]
        else:
            mappings_df.loc[i, "sc_locations"] = r.target
            mappings_df.loc[i, "sc_target"] = r.target
    return mappings_df

def get_average_intensities_tsp(ls_): #warping
    n = len(ls_)
    sample_img = imread(f"{sampled_intensity_dir}/{ls_[0]}_protein.png")
    intensities = np.zeros(sample_img.shape)
    for img_id in ls_:
        try:
            pilr = imread(f"{sampled_intensity_dir}/{img_id}_protein.png")
        except:
            print(f"{sampled_intensity_dir}/{img_id}_protein.png reading err")
        try:
            thres = threshold_otsu(pilr)
            #thres = np.percentile(pilr.ravel(), 90)
        except:
            thres = 0
        pilr = 1*(pilr > thres).astype("float64")
        intensities += pilr / n
    return intensities

def get_average_intensities_cr(ls_): #concentric rings
    n = len(ls_)
    intensities = np.zeros((31,256))
    for img_id in ls_:
        try:
            pilr = np.load(f"{sampled_intensity_dir}/{img_id}_protein.npy")
        except:
            print(f"{sampled_intensity_dir}/{img_id}_protein.npy reading err")
        try:
            thres = threshold_otsu(pilr)
        except:
            thres = 0
        pilr = (pilr > thres).astype("float64")
        intensities += pilr / n
    return intensities

def get_mask(file_path=f"Avg_cell.npz", shape_=(336, 699)):
    avgcell = np.load(file_path)
    ix_c = avgcell["ix_c"]
    iy_c = avgcell["iy_c"]
    min_x = np.min(ix_c)
    min_y = np.min(iy_c)
    nu_centroid = [0,0]
    nu_centroid[0] = -min_x
    nu_centroid[1] = -min_y
    ix_c -= min_x
    iy_c -= min_y
    from skimage.draw import polygon
    img = np.zeros(shape_)
    rr, cc = polygon(ix_c, iy_c, img.shape)
    img[rr, cc] = 1
    id_keep = np.where(img.flatten()==1)[0]
    return id_keep, img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_line", help="principle component", type=str)
    args = parser.parse_args()
    intensity_sampling_concentric_ring = False
    intensity_warping = True
    import configs.config as cfg
    
    cell_line = args.cell_line #cfg.CELL_LINE #args.cell_line
    project_dir = os.path.join(os.path.dirname(cfg.PROJECT_DIR), cell_line) #cfg.PROJECT_DIR#os.path.join(os.path.dirname(cfg.PROJECT_DIR), cell_line)

    log_dir = f"{project_dir}/logs"
    fft_dir = f"{project_dir}/fftcoefs/{cfg.ALIGNMENT}"
    fft_path = os.path.join(fft_dir, f"fftcoefs_{cfg.N_COEFS}.txt")
    shape_mode_path = f"{project_dir}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}"
    if intensity_sampling_concentric_ring:
        avg_organelle_dir = f"{project_dir}/matrix_protein_avg"
        sampled_intensity_dir = f"{project_dir}/sampled_intensity_bin"
    if intensity_warping:
        avg_organelle_dir = f"{project_dir}/warps_protein_avg_otsu" 
        sampled_intensity_dir = f"{project_dir}/warps" 
    
    os.makedirs(avg_organelle_dir, exist_ok=True)
    cellline_meta = os.path.join(project_dir, os.path.basename(cfg.META_PATH).replace(".csv", "_splitVesiclesPCP.csv"))
    print(cellline_meta)
    if os.path.exists(cellline_meta):
        mappings = pd.read_csv(cellline_meta)
    else:
        mappings = pd.read_csv(cfg.META_PATH)
        mappings = mappings[mappings.atlas_name == cell_line]
        mappings["cell_idx"] = [idx.split("_", 1)[1] for idx in mappings.id]
        mappings = unmerge_label(mappings)
        mappings.to_csv(cellline_meta, index=False)
        print(mappings.sc_target.value_counts())

    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json", "r")
    cells_assigned = json.load(f)
    #merged_bins = [[0], [1], [2], [3], [4], [5], [6]]
    merged_bins = [[0], [3], [6]]
    
    # Average organelles
    lines = []
    lines.append(["PC","Organelle", "bin","n_cells"])
    for PC in [1]:#np.arange(1,7):
        pc_cells = cells_assigned[f"PC{PC}"]
        for org in cfg.ORGANELLES:
            for i, bin_ in enumerate(merged_bins):
                ls = [pc_cells[b] for b in bin_]
                ls = helpers.flatten_list(ls)
                ls = [os.path.basename(l).replace(".npy", "") for l in ls]
                df_sl = mappings[mappings.cell_idx.isin(ls)]
                #ls_ = df_sl[df_sl.sc_target == org].cell_idx.to_list()
                ls_ = df_sl[df_sl.locations == org].cell_idx.to_list()

                if len(ls_)==0:
                    print(f"{org}: Found {len(ls_)}, eg: {ls_[:3]}")
                    continue
                n0 = len(ls_)
                lines.append([f"PC{PC}", org, bin_[0], n0])
                if os.path.exists(f"{avg_organelle_dir}/PC{PC}_{org}_b{bin_[0]}.png"):
                   continue
                if len(ls_) < 5:
                    print(f"{org} has less than 5 cells ({len(ls_)}) -> move on")
                    continue
                if n0 > 1000:
                    import random
                    ls_ = random.sample(ls_, 1000)
                if intensity_sampling_concentric_ring:
                    intensities = get_average_intensities_cr(ls_)
                    np.save(f"{avg_organelle_dir}/PC{PC}_{org}_b{bin_[0]}.npy", intensities)
                if intensity_warping:
                    intensities = get_average_intensities_tsp(ls_)
                    intensities = (intensities*255).astype('uint8')
                    imwrite(f"{avg_organelle_dir}/PC{PC}_{org}_b{bin_[0]}.png", intensities)
                print(f"PC{PC}_{org}_b{bin_[0]}.png {len(ls_)} cells. Accumulated: {intensities.max()}, {intensities.dtype}")
                #print(org, intensities.sum(axis=1))
    df = pd.DataFrame(lines)
    df.to_csv(f"{avg_organelle_dir}/organelle_distr.csv", index=False)
    shape_ = imread(glob.glob(f"{avg_organelle_dir}/*.png")[0]).shape
    id_keep, mask = get_mask(file_path=f"{shape_mode_path}/Avg_cell.npz", shape_=shape_)
    # Organelle heatmap through shapespace
    for PC in [1]:#np.arange(1,7):
        for i, bin_ in enumerate(merged_bins):
            if len(bin_) == 1:
                b = bin_[0]
                images = {}
                for org in cfg.ORGANELLES:
                    if intensity_sampling_concentric_ring:
                        ch = np.load(f"{avg_organelle_dir}/PC{PC}_{org}_b{b}.npy")
                    if intensity_warping:
                        try:
                            ch = imread(f"{avg_organelle_dir}/PC{PC}_{org}_b{b}.png")
                        except:
                            print(f"{avg_organelle_dir}/PC{PC}_{org}_b{b}.png not found, defaulting it to 0")
                            ch = np.zeros(shape_) #np.array([0])
                    images[org] = ch
            
            ssim_scores = correlation(images, pearsonr, id_keep) #structural_similarity)
            ssim_df = pd.DataFrame(ssim_scores, columns=list(images.keys()))
            ssim_df.index = list(images.keys())
            print(ssim_df)
            ssim_df.to_csv(f"{avg_organelle_dir}/PC{PC}_bin{b}_pearsonr_df.csv")
            plt.figure()
            sns.heatmap(ssim_df, cmap="RdBu", vmin=-1, vmax=1)
            plt.savefig(f"{avg_organelle_dir}/PC{PC}_bin{b}_pearsonr.png")
