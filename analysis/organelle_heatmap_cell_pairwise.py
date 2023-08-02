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
import random

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

def load_intensities(ls_, sampled_intensity_dir, id_keep):
    intensities = []
    for img_id in ls_:
        pilr = imread(f"{sampled_intensity_dir}/{img_id}_protein.png")
        thres = threshold_otsu(pilr)
        pilr = 1*(pilr > thres)
        intensities += [pilr.flatten()[id_keep]]
    return np.array(intensities)

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
    return id_keep

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
        avg_organelle_dir = f"{project_dir}/warps_protein_avg_cell_pairwise" 
        sampled_intensity_dir = f"{project_dir}/warps" 
    
    id_keep = get_mask(file_path=f"{shape_mode_path}/Avg_cell.npz")

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
    merged_bins = [[0], [3], [6]]
     
    # Organelle heatmap through shapespace
    for PC in [1]:#np.arange(1,7):
        pc_cells = cells_assigned[f"PC{PC}"]
        for bin_ in merged_bins:
            if len(bin_) == 1:
                b = bin_[0]
                ls = [pc_cells[b] for b in bin_]
                ls = helpers.flatten_list(ls)
                ls = [os.path.basename(l).replace(".npy", "") for l in ls]
                df_sl = mappings[mappings.cell_idx.isin(ls)]
                cor_mat = np.zeros((len(cfg.ORGANELLES), len(cfg.ORGANELLES)))
                for i, org1 in enumerate(cfg.ORGANELLES):
                    ls_i = df_sl[df_sl.sc_target == org1].cell_idx.to_list()
                    if len(ls_i)<5:
                        continue
                    ls_ = (random.sample(ls_i, 100) if len(ls_i)>100 else ls_i)
                    intensities_1 = load_intensities(ls_, sampled_intensity_dir, id_keep)
                    n = len(ls_)
                    print(intensities_1.shape)
                    for j, org2 in enumerate(cfg.ORGANELLES):
                        ls_j = df_sl[df_sl.sc_target == org2].cell_idx.to_list()
                        print(f"{org1}: {len(ls_i)}, {org2}: {len(ls_j)}")
                        if len(ls_j)<5:
                            continue
                        ls_ = (random.sample(ls_j, 100) if len(ls_j)>100 else ls_j)
                        intensities_2 = load_intensities(ls_, sampled_intensity_dir, id_keep)
                        tmp = np.corrcoef(intensities_1, intensities_2, rowvar=True)                        
                        cor_mat[i, i] = tmp[:n, :n].mean()
                        cor_mat[i, j] = tmp[:n, n:].mean()
                        #cor_mat[j, i] = tmp[n:, :n].mean()
                        #cor_mat[j, j] = tmp[n:, n:].mean()
                        print(tmp[:n, n:].mean(), tmp[n:, :n].min(), tmp.min())

                    cor_mat = pd.DataFrame(cor_mat, columns=cfg.ORGANELLES, index=cfg.ORGANELLES)
                    #print(cor_mat)
                    plt.figure()
                    sns.heatmap(cor_mat, cmap="RdBu", vmin=-0.5, vmax=1)
                    plt.savefig(f"{avg_organelle_dir}/PC{PC}_bin{b}_pearsonr.png")
            breakme
