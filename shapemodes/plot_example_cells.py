import os
import json
import pandas as pd
import numpy as np
import sys

sys.path.append("..")
from utils import plotting

cell_line = "U-2 OS"
project_dir = f"/data/2Dshapespace/{cell_line.replace(' ','_')}"
fft_dir = f"{project_dir}/fftcoefs"
n_coef = 128
fft_path = os.path.join(fft_dir, f"fftcoefs_{n_coef}.txt")

shape_mode_path = f"{project_dir}/shapemode/{cell_line.replace(' ','_')}/ratio8"
f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json", "r")
cells_assigned = json.load(f)
n_cells = dict()
for k, v in cells_assigned.items():
    n_cells[k] = [len(b) for b in v]

n_cells = pd.DataFrame(n_cells)
n_cells

pcs = list(cells_assigned.keys())
for pc in pcs:
    bin_links = cells_assigned[pc]
    plotting.plot_example_cells(
        bin_links,
        n_coef=128,
        cells_per_bin=10,
        shape_coef_path=fft_path,
        save_path=f"{shape_mode_path}/{pc}_example_cells.png",
    )
