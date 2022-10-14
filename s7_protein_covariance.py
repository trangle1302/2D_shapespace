import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from utils import helpers
import glob
import matplotlib.pyplot as plt
from utils.parameterize import get_coordinates
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--PC", help="shape mode",
                    type=str)
    args = parser.parse_args()
    print(args.PC)

    n_coef = 128
    cell_line = "U-2 OS"
    project_dir = f"/scratch/users/tle1302/2Dshapespace/{cell_line.replace(' ','_')}"
    log_dir = f"{project_dir}/logs"
    fftcoefs_dir = f"{project_dir}/fftcoefs"
    fft_path = os.path.join(fftcoefs_dir,f"fftcoefs_{n_coef}.txt")
    shape_mode_path = f"{project_dir}/shapemode/{cell_line.replace(' ','_')}/ratio8"

    sampled_intensity_dir = f"{project_dir}/sampled_intensity"

    mappings = pd.read_csv("/scratch/users/tle1302/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border.csv")
    id_with_intensity = glob.glob(f"{sampled_intensity_dir}/*.npy")
    mappings["Link"] =[f"{sampled_intensity_dir}/{id.split('_',1)[1]}_protein.npy" for id in mappings.id]
    mappings = mappings[mappings.Link.isin(id_with_intensity)]
    print(mappings.target.value_counts())

    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json")
    cells_assigned = json.load(f)
    print(cells_assigned.keys())
    save_dir = f"{project_dir}/shapemode/covar"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    meta = []
    merged_bins = [[0,1,2,3],[4,5,6],[7,8,9,10]]
    PC = args.PC
    pc_cells = cells_assigned[f"PC{PC}"]
    if True: #for PC, pc_cells in cells_assigned.items():
        shape = (21, n_coef*2)
        intensities_pcX = []
        counts = []
        for i, bin_ in enumerate(merged_bins):
            intensities = []
            ensembl_ids = []
            ls = [pc_cells[b] for b in bin_]
            ls = helpers.flatten_list(ls)
            ls = [f"{sampled_intensity_dir}/{os.path.basename(l)}".replace(".npy","_protein.npy") for l in ls]
            print(ls[:3])
            print(mappings.Link)
            df_sl = mappings[mappings.Link.isin(ls)]
            print(f"PC{PC}: Number of cells {df_sl.shape[0]}, Number of gene: {df_sl.ensembl_ids.nunique()}")
            for _, row in df_sl.iterrows():
                intensity = np.load(row.Link)
                dummy_threshold = intensity.max() // 3
                intensity = np.where(intensity > dummy_threshold, 1, 0)
                intensities += [intensity.flatten()]
                ensembl_ids += [row.ensembl_ids]
            intensities = pd.DataFrame(intensities)
            intensities["ensembl_ids"] = ensembl_ids
            intensities = intensities.groupby('ensembl_ids').agg("mean")
            intensities.to_csv(f"{save_dir}/PC{PC}_{i}_intensities.csv")
            """
            #intensities_pcX += [np.nanmean(intensities, axis=0).reshape(intensity.shape)]
            covar_mat = np.corrcoef(np.array(intensities.drop(["ensembl_ids"], axis=1)))
            covar_mat = pd.DataFrame(covar_mat)
            covar_mat.column = ensembl_ids
            covar_mat.index = ensembl_ids
            covar_mat.to_csv(f"{save_dir}/{}.csv")
            """
            covar_mat = intensities.corr()
            covar_mat.to_csv(f"{save_dir}/PC{PC}_{i}.csv")
    #np.corrcoef(xarr, yarr, rowvar=False) #row-wise correlation
    #np.corrcoef(xarr, yarr, rowvar=False) #column-wise correlation