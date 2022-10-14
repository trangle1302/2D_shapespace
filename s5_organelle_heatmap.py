import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from utils import plotting, parameterize
import glob
import matplotlib.pyplot as plt
from utils.parameterize import get_coordinates
import json

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

    #mappings = pd.read_csv(f"/data/kaggle-dataset/publicHPA_umap/results/webapp/pHPA10000_15_0.1_euclidean_ilsc_2d_bbox_nobordercells.csv")
    mappings = pd.read_csv("/scratch/users/tle1302/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border.csv")
    #mappings = pd.read_csv("/scratch/users/tle1302/pHPA10000_15_0.1_euclidean_ilsc_2d_bbox_nobordercells.csv")
    print(mappings.columns)
    id_with_intensity = glob.glob(f"{sampled_intensity_dir}/*.npy")
    mappings["Link"] =[f"{sampled_intensity_dir}/{id.split('_',1)[1]}_protein.npy" for id in mappings.id]
    mappings = mappings[mappings.Link.isin(id_with_intensity)]
    print(mappings.target.value_counts())

    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json")
    cells_assigned = json.load(f)
    print(cells_assigned.keys())
    save_dir = f"{project_dir}/shapemode/organelle2"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    meta = []
    org = args.org
    df_sl_Label = mappings[mappings.target == org]
    
    for PC, pc_cells in cells_assigned.items():
        if os.path.exist(f"{save_dir}/{org}_{PC}_intensity.npz"):
            continue
        print(org, PC, len(pc_cells), len(pc_cells[0]))
        shape = (21, n_coef*2)
        intensities_pcX = []
        counts = []
        for ls in pc_cells:
            intensities = []
            i= 0
            for l in ls:
                l = str(sampled_intensity_dir) + "/"+ Path(l).stem + "_protein.npy"
                if l in list(df_sl_Label.Link):
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
        meta += [[org]+ counts]
        intensities_pcX = np.array(intensities_pcX)
        print(intensities_pcX.shape)
        np.save(f"{save_dir}/{org}_{PC}_intensity", intensities_pcX)
        #pm.protein_intensities = intensities_pcX/intensities_pcX.max()
        #pm.plot_protein_through_shape_variation_gif(PC, title=org, dark=True, save_dir=f"{project_dir}/shapemode/organelle")
        #data = np.load(f"{shape_mode_path}/shapevar_{PC}.npz")
        print(data["nuc"].shape, data["mem"].shape)
        #x_,y_ = parameterize.get_coordinates(data["nuc"], data["mem"], [0,0], n_isos = [10,10], plot=False)
        #print(f"saving to {save_dir}")
        #plotting._plot_protein_through_shape_variation_gif(PC, data["nuc"], data["mem"], intensities_pcX/intensities_pcX.max(), title=org, dark=True, save_dir=save_dir)
    meta = pd.DataFrame(meta)
    meta.columns = ["org"] +["".join(("n_bin",str(i))) for i in range(11)]
    print(meta)
    meta.to_csv(f"{save_dir}/{org}cells_per_bin.csv", index=False)
