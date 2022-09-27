import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from utils import plotting
import glob
import matplotlib.pyplot as plt
from utils.parameterize import get_coordinates
import json

LABEL_NAMES = {
  0: 'Nucleoplasm',
  1: 'Nuclear membrane',
  2: 'Nucleoli',
  3: 'Nucleoli fibrillar center',
  4: 'Nuclear speckles',
  5: 'Nuclear bodies',
  6: 'Endoplasmic reticulum',
  7: 'Golgi apparatus',
  8: 'Intermediate filaments',
  9: 'Actin filaments',
  10: 'Microtubules',
  11: 'Mitotic spindle',
  12: 'Centrosome',
  13: 'Plasma membrane',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'Vesicles and punctate cytosolic patterns',
  18: 'Negative',
}

all_locations = dict((v, k) for k,v in LABEL_NAMES.items())

def avg_matrix(bins, intensity_matrix):
    df = []
    #for 
    return df 

if __name__ == "__main__":
    n_coef = 128
    cell_line = "U-2 OS"
    save_dir = f"/data/2Dshapespace/{cell_line.replace(' ','_')}/cell_masks"
    fftcoefs_dir = Path(f"/data/2Dshapespace/{cell_line.replace(' ','_')}/fftcoefs")
    sampled_intensity_dir = Path(f"/data/2Dshapespace/{cell_line.replace(' ','_')}/sampled_intensity")

    mappings = pd.read_csv(f"/data/kaggle-dataset/publicHPA_umap/results/webapp/pHPA10000_15_0.1_euclidean_ilsc_2d_bbox_nobordercells.csv")
    #print(mappings.target.value_counts())
    print(mappings.columns)
    id_with_intensity = glob.glob(f"{sampled_intensity_dir}/*.npy")
    mappings["Link"] =[f"/data/2Dshapespace/U-2_OS/sampled_intensity/{id.split('_',1)[1]}_protein.npy" for id in mappings.id]
    mappings = mappings[mappings.Link.isin(id_with_intensity)]
    print(mappings.target.value_counts())

    f = open('cells_assigned_to_pc_bins.json')
    cells_assigned = json.load(f)

    for PC, pc_cells in cells_assigned.items():
            print("xxxx", len(pc_cells), len(pc_cells[0]))
            shape = (21,n_coef*2)
            intensities_pcX = []
            counts = []