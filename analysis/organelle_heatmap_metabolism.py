import os
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
#from utils import helpers
from organelle_heatmap import get_mask, get_average_intensities_cr, get_average_intensities_tsp, correlation
from organelle_heatmap_helpers import get_cells, plot_pilr_collage, plot_image_collage, get_heatmap

import configs.config as cfg



if __name__ == "__main__":
    intensity_sampling_concentric_ring = False #True
    intensity_warping = True
    cell_line = 'U2OS' #'Hep-G2' #'U2OS' # 'S-BIAD34'

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

    avg_organelle_dir = f"{avg_organelle_dir}/metabolism"
    os.makedirs(avg_organelle_dir, exist_ok=True)
    #mappings = pd.read_csv('/data/HPA-IF-images/IF-image.csv')
    #mappings = mappings[['gene_names','antibody','ensembl_ids','locations']].dropna().drop_duplicates()
    #mappings = mappings[mappings.atlas_name == 'U2OS']
    cellline_meta = os.path.join(project_dir, os.path.basename(cfg.META_PATH).replace(".csv", "_splitVesiclesPCP.csv"))
    mappings = pd.read_csv(cellline_meta)
    sc_stats = pd.read_csv(f"{project_dir}/single_cell_statistics.csv")
    #sc_stats["cell_idx"] = ['_'.join([r.image_id, str(r.cell_id)]) for _,r in sc_stats.iterrows()]
    sc_stats["cell_idx"] = sc_stats.apply(lambda r: f'{r.image_id}_{r.cell_id}', axis=1)
    mappings = mappings.merge(sc_stats, on='cell_idx', how='left')
    human1 = pd.read_csv("/home/trangle/2D_shapespace/20230111_Human1_ProteinMapping.csv")
    human1['PathwayGroup'] = human1['PathwayGroup'].str.replace(' ', '-')
    human1['PathwayGroup'] = human1['PathwayGroup'].str.replace('/', '-')
    human1['Pathway'] = human1['Pathway'].str.replace(' ', '-')
    human1['Pathway'] = human1['Pathway'].str.replace('/', '-')
    mappings['Gene_name'] = [f.split(',') for f in mappings.gene_names]
    mappings = mappings.explode('Gene_name')
    mappings = mappings.merge(human1, on='Gene_name', how='left') # Keeping only metabolic proteins

    shape_mode_path = f"{project_dir}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}"
    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json", "r")
    cells_assigned = json.load(f)
    for pc_name in list(cells_assigned.keys())[:6]:
        mappings[pc_name] = np.nan
        for i in range(7):
            ls = get_cells(cells_assigned, PC=pc_name, bin_=[i])
            ls = [os.path.basename(l).replace(".npy", "") for l in ls]
            mappings.loc[mappings.cell_idx.isin(ls), pc_name] = i

    mappings.head()
    print(mappings.PathwayGroup.value_counts())
    mappings['Protein_nu_mean'] = mappings.Protein_nu_sum / mappings.nu_area
    mappings['Protein_cytosol_mean'] = (mappings.Protein_cell_sum - mappings.Protein_nu_sum)/(mappings.cell_area_x - mappings.nu_area)

    classes = mappings[mappings.PathwayGroup=='Glycan-biosynthesis-and-metabolism'].Pathway.unique()
    for pathwayclass in classes:
        org_dstr = get_heatmap(mappings[mappings.PathwayGroup=='Glycan-biosynthesis-and-metabolism'], cells_assigned, pathway_group = pathwayclass)#, pathway = 'Pathway')