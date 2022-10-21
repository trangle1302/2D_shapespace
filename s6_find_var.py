import os
import numpy as np
import pandas as pd
import argparse
import glob
#import matplotlib.pyplot as plt
import json
import gseapy


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

    mappings = pd.read_csv("/scratch/users/tle1302/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border.csv")
    id_with_intensity = glob.glob(f"{sampled_intensity_dir}/*.npy")
    mappings["Link"] =[f"{sampled_intensity_dir}/{id.split('_',1)[1]}_protein.npy" for id in mappings.id]
    mappings = mappings[mappings.Link.isin(id_with_intensity)]
    print(mappings.target.value_counts())
    print(mappings.shape)

    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json")
    cells_assigned = json.load(f)
    print(cells_assigned.keys())
    save_dir = f"{project_dir}/shapemode/gsea"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    meta = []
    org = args.org
    df_sl_Label = mappings[mappings.target == org]
    print(df_sl_Label.shape)
    
    databases = ['GO_Biological_Process_2021', 'GO_Cellular_Component_2021', 'GO_Molecular_Function_2021', 'WikiPathway_2021_Human', 'KEGG_2021_Human']
    
    for PC, pc_cells in cells_assigned.items():
        shape = (21, n_coef*2)
        intensities_pcX = []
        for i, ls in enumerate(pc_cells):
            print(ls)
            print(mappings.Link)
            gene_list = mappings[mappings.Link.isin(ls)].ensembl_ids.unique()
            print(f"{org}-PC{PC}: Number of cells {len(pc_cells[i])}, Number of gene: {len(gene_list)}")
            enr = gseapy.enrichr(gene_list=gene_list, description='pathway', gene_sets=databases,
            outdir=f'{save_dir}/{PC}', background='hsapiens_gene_ensembl', cutoff=0.1, format='pdf')
            print(enr.results.head(5))

    #np.corrcoef(xarr, yarr, rowvar=False) #row-wise correlation
    #np.corrcoef(xarr, yarr) #column-wise correlation