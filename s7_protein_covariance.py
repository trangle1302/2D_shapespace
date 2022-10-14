import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from utils import helper
import glob
import matplotlib.pyplot as plt
from utils.parameterize import get_coordinates
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", help="Organelle class",
                    type=str)
    args = parser.parse_args()
    print(args.org)

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
    for PC, pc_cells in cells_assigned.items():
        print(org, PC, len(pc_cells), len(pc_cells[0]))
        shape = (21, n_coef*2)
        intensities_pcX = []
        counts = []
        for bin_ in merged_bins:
            intensities = []
            ls = [pc_cells[b] for b in bin_]
            ls = helper.flatten_list(ls)
            df_sl_Label = mappings[mappings.Link.isin(ls)]
        for ls in pc_cells:
            intensities = []
            i= 0
            
            for l in ls:
                intensity = np.load(l)
                dummy_threshold = intensity.max() // 3
                intensity = np.where(intensity > dummy_threshold, 1, 0)
                intensities += [intensity.flatten()]
                i +=1
            counts += [i]
            if len(intensities) == 0:
                print(f'No cell sample at this bin for {org}')
                intensities_pcX += [np.zeros(shape)]
            else:
                print(len(intensities))
                intensities_pcX += [np.nanmean(intensities, axis=0).reshape(intensity.shape)]
        print(counts) 
    np.cov(intensities)  
    np.corrcoef(xarr, yarr, rowvar=False) #row-wise correlation
    np.corrcoef(xarr, yarr, rowvar=False) #column-wise correlation