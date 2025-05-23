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
from organelle_heatmap import unmerge_label, get_mask

def load_intensities(ls_, sampled_intensity_dir, id_keep):
    intensities = []
    for img_id in ls_:
        pilr = imread(f"{sampled_intensity_dir}/{img_id}_protein.png")
        thres = threshold_otsu(pilr)
        pilr = 1*(pilr > thres)
        intensities += [pilr.flatten()[id_keep]]
    return np.array(intensities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_line", help="Cell Line", type=str)
    args = parser.parse_args()
    intensity_sampling_concentric_ring = False
    intensity_warping = True
    import configs.config as cfg
    
    # If not specified, use the cell line in config file
    if args.cell_line is None:
        cell_line = cfg.CELL_LINE
    else:
        cell_line = args.cell_line
    project_dir = os.path.join(os.path.dirname(cfg.PROJECT_DIR), cell_line)
    print('Working dir: ', project_dir)
    
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
    
    id_keep, mask = get_mask(file_path=f"{shape_mode_path}/Avg_cell.npz")

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
                for i, org1 in enumerate(cfg.ORGANELLES_FULLNAME):
                    #ls_i = df_sl[df_sl.sc_target == org1].cell_idx.to_list()
                    ls_i = df_sl[df_sl.locations == org1].cell_idx.to_list()
                    if len(ls_i)<5:
                        continue
                    ls_ = (random.sample(ls_i, 100) if len(ls_i)>100 else ls_i)
                    intensities_1 = load_intensities(ls_, sampled_intensity_dir, id_keep)
                    n = len(ls_)
                    print(intensities_1.shape)
                    for j, org2 in enumerate(cfg.ORGANELLES_FULLNAME):
                        #ls_j = df_sl[df_sl.sc_target == org2].cell_idx.to_list()
                        ls_j = df_sl[df_sl.locations == org1].cell_idx.to_list()
                        print(f"{org1}: {len(ls_i)}, {org2}: {len(ls_j)}")
                        if len(ls_j)<5:
                            continue
                        ls_ = (random.sample(ls_j, 100) if len(ls_j)>100 else ls_j)
                        intensities_2 = load_intensities(ls_, sampled_intensity_dir, id_keep)
                        print(org1, org2, np.corrcoef(intensities_1[0], intensities_2[0])[0,1])
                        print(org1, org2, np.corrcoef(intensities_1[1], intensities_2[1])[0,1])
                        print(org1, org2, np.corrcoef(intensities_1[5], intensities_2[5])[0,1])
                        tmp = np.corrcoef(intensities_1, intensities_2, rowvar=True)                        
                        cor_mat[i, i] = tmp[:n, :n].mean()
                        cor_mat[i, j] = tmp[:n, n:].mean()
                        #cor_mat[j, i] = tmp[n:, :n].mean()
                        #cor_mat[j, j] = tmp[n:, n:].mean()
                        print(tmp[:n, n:].mean(), tmp[n:, :n].min())
                        plt.figure()
                        sns.heatmap(tmp, cmap="RdBu", vmin=-1, vmax=1)
                        plt.savefig(f"{avg_organelle_dir}/PC{PC}_bin{b}_{org1}{len(intensities_1)}_{org2}{len(intensities_2)}_pearsonr.png")
                        print('##############################################')

                    cor_mat = pd.DataFrame(cor_mat, columns=cfg.ORGANELLES, index=cfg.ORGANELLES)
                    #print(cor_mat)
                    plt.figure()
                    sns.heatmap(cor_mat, cmap="RdBu", vmin=-0.5, vmax=1)
                    plt.savefig(f"{avg_organelle_dir}/PC{PC}_bin{b}_pearsonr.png")
            breakme
