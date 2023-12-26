import os
import numpy as np
import pandas as pd
import argparse
import glob
import sys
sys.path.append("..")
import json
import gseapy
import configs.config as cfg

if __name__ == "__main__":    
    log_dir = f"{cfg.PROJECT_DIR}/logs"
    fftcoefs_dir = f"{cfg.PROJECT_DIR}/fftcoefs"
    fft_path = os.path.join(fftcoefs_dir, f"fftcoefs_{cfg.N_COEFS}.txt")
    shape_mode_path = f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}"
    sampled_intensity_dir = f"{cfg.PROJECT_DIR}/sampled_intensity_bin"
    id_with_intensity = glob.glob(f"{sampled_intensity_dir}/*.npy")

    mappings = pd.read_csv(f"{cfg.PROJECT_DIR}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border_splitVesiclesPCP.csv")
    mappings["Link"] = [sampled_intensity_dir + "/"+ f.split("_",1)[1] + "_protein.npy" for f in mappings.id]
    
    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json")
    cells_assigned = json.load(f)
    print(cells_assigned.keys())
    save_dir = f"{shape_mode_path}/gsea"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    meta = []
    merged_bins = [[0, 1, 2], [3], [4, 5, 6]]
    databases = [
        "GO_Biological_Process_2021",
        "GO_Cellular_Component_2021",
        "GO_Molecular_Function_2021",
        "WikiPathway_2021_Human",
        "KEGG_2021_Human",
        "CORUM"
    ]

    for PC, pc_cells in cells_assigned.items():
        """
        shape = (21, n_coef*2)
        intensities_pcX = []
        for i, ls in enumerate(pc_cells):
            ls = [f"{sampled_intensity_dir}/{os.path.basename(l)}".replace(".npy","_protein.npy") for l in ls]
            #print(ls[:3],mappings.Link)
            gene_list = mappings[mappings.Link.isin(ls)].ensembl_ids.unique()
            gene_list = [g.split(",")[0] for g in gene_list]
            #gene_list = helpers.flatten_list(gene_list)
            print(gene_list[:3])
            print(f"PC{PC}: Number of cells {len(pc_cells[i])}, Number of gene: {len(gene_list)}")
            enr = gseapy.enrichr(gene_list=list(gene_list), gene_sets=databases, organism="human",
                outdir=f'{save_dir}/{PC}', background='hsapiens_gene_ensembl', cutoff=0.1, format='pdf')
            print(enr)#.results.head(5))
        """
        for i in range(3):
            try:
                f = open(
                    f"{cfg.PROJECT_DIR}/covar/{PC}_{i}_cluster_assignation.json",
                    "r",
                )
                clusters = json.load(f)
                for cluster in np.unique(clusters["assignation"]):
                    gene_list = np.array(clusters["ensembl_ids"])[
                        np.where(np.array(clusters["assignation"]) == cluster)[0].astype(
                            "uint"
                        )
                    ]
                    gene_list = [g.split(",")[0] for g in gene_list]
                    print(gene_list[:10])
                    print(f"PC{PC}-bin{i}: Number of gene: {len(gene_list)}")
                    enr = gseapy.enrichr(
                        gene_list=list(gene_list),
                        gene_sets=databases,
                        organism="human",
                        outdir=f"{save_dir}/{PC}_{i}",
                        background="hsapiens_gene_ensembl",
                        cutoff=0.1,
                        format="pdf",
                    )
                    print(PC, i, enr.results.head(5))
            except:
                print(f"Can't find {cfg.PROJECT_DIR}/covar/{PC}_{i}_cluster_assignation.json")
    # np.corrcoef(xarr, yarr, rowvar=False) #row-wise correlation
    # np.corrcoef(xarr, yarr) #column-wise correlation
