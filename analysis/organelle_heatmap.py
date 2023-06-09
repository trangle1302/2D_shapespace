import os
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import threshold_minimum
from skimage.metrics import structural_similarity
from scipy.stats import pearsonr
import json
from utils import helpers
import argparse

def correlation(value_dict, method_func):
    cor_mat = np.zeros((len(value_dict), len(value_dict)))
    for i, (k1, v1) in enumerate(value_dict.items()):
        for j, (k2, v2) in enumerate(value_dict.items()):
            print(k1, k2, method_func(v1.flatten(),v2.flatten()))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_line", help="principle component", type=str)
    args = parser.parse_args()

    import configs.config as cfg
    
    cell_line = cfg.CELL_LINE #args.cell_line
    project_dir = cfg.PROJECT_DIR#os.path.join(os.path.dirname(cfg.PROJECT_DIR), cell_line)

    log_dir = f"{project_dir}/logs"
    fft_dir = f"{project_dir}/fftcoefs/{cfg.ALIGNMENT}"
    fft_path = os.path.join(fft_dir, f"fftcoefs_{cfg.N_COEFS}.txt")
    shape_mode_path = f"{project_dir}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}"
    avg_organelle_dir = f"{project_dir}/matrix_protein_avg"
    os.makedirs(avg_organelle_dir, exist_ok=True)
    sampled_intensity_dir = f"{project_dir}/sampled_intensity"

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
    merged_bins = [[0], [1], [2], [3], [4], [5], [6]]
    # Panel 1: Organelle through shapespace
    

    for PC in np.arange(1,7):
        lines = []
        pc_cells = cells_assigned[f"PC{PC}"]
        for org in cfg.ORGANELLES:
            for i, bin_ in enumerate(merged_bins):
                #if os.path.exists(f"{avg_organelle_dir}/PC{PC}_{org}_b{bin_[0]}.npy"):
                #    continue
                ls = [pc_cells[b] for b in bin_]
                ls = helpers.flatten_list(ls)
                ls = [os.path.basename(l).replace(".npy", "") for l in ls]
                df_sl = mappings[mappings.cell_idx.isin(ls)]
                ls_ = df_sl[df_sl.sc_target == org].cell_idx.to_list()
                print(f"Found {len(ls_)}, eg: {ls[:3]}")
                intensities = np.zeros((31,256))
                n = len(ls_)
                for img_id in ls_: #tqdm(ls_, desc=f"{PC}_bin{bin_[0]}_{org}"):    
                    pilr = np.load(f"{sampled_intensity_dir}/{img_id}_protein.npy")
                    try:
                        thres = threshold_minimum(pilr)
                    except:
                        thres = 0
                    print(thres)
                    pilr = (pilr > thres).astype("float64")
                    intensities += pilr / n
                print("Accumulated: ", intensities.max(), intensities.dtype, "Addition: ", pilr.max(), pilr.dtype,  (pilr / len(ls_)).max())
                np.save(f"{avg_organelle_dir}/PC{PC}_{org}_b{bin_[0]}.npy", intensities)
                lines += [f"PC{PC}", org, bin_[0], n]
    #df = pd.DataFrame(lines)
    #print(df)
    #df.to_csv(f"{avg_organelle_dir}/organelle_distr.csv", index=False)
    
    # Panel 2: Organelle heatmap through shapespace
    for PC in np.arange(1,7):
        for b in np.arange(11):
            images = {}
            for i, bin_ in enumerate(merged_bins):
                if len(bin_) == 1:
                    b = bin_[0]
                    images = {}
                    for org in cfg.ORGANELLES:
                        images[org] = np.load(f"{avg_organelle_dir}/PC{PC}_{org}_b{b}.npy")

            ssim_scores = correlation(images, pearsonr)#structural_similarity)
            ssim_df = pd.DataFrame(ssim_scores, columns=list(images.keys()))
            ssim_df.index = list(images.keys())
            ssim_df.to_csv(f"{avg_organelle_dir}/PC{PC}_bin{b}_pearsonr_df.csv")
