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


LABEL_TO_ALIAS = {
    0: "Nucleoplasm",
    1: "NuclearM",
    2: "Nucleoli",
    3: "NucleoliFC",
    4: "NuclearS",
    5: "NuclearB",
    6: "EndoplasmicR",
    7: "GolgiA",
    8: "IntermediateF",
    9: "ActinF",
    10: "Microtubules",
    11: "MitoticS",
    12: "Centrosome",
    13: "PlasmaM",
    14: "Mitochondria",
    15: "Aggresome",
    16: "Cytosol",
    17: "VesiclesPCP",
    19: "Negative",
    19: "Multi-Location",
}

all_locations = dict((v, k) for k, v in LABEL_TO_ALIAS.items())
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", help="Organelle class", type=str)
    args = parser.parse_args()
    print(args.org)
    
    log_dir = f"{cfg.PROJECT_DIR}/logs"
    fftcoefs_dir = f"{cfg.PROJECT_DIR}/fftcoefs"
    fft_path = os.path.join(fftcoefs_dir, f"fftcoefs_{cfg.N_COEFS}.txt")
    shape_mode_path = f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}"

    sampled_intensity_dir = f"{cfg.PROJECT_DIR}/sampled_intensity"
    id_with_intensity = glob.glob(f"{sampled_intensity_dir}/*.npy")

    mappings = pd.read_csv(cfg.META_PATH)
    mappings["Link"] = [
        f"{sampled_intensity_dir}/{id.split('_',1)[1]}_protein.npy"
        for id in mappings.id
    ]
    mappings = mappings[mappings.Link.isin(id_with_intensity)]
    print(mappings.target.value_counts())
    print(mappings.shape)

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
            f = open(
                f"{cfg.PROJECT_DIR}/shapemode/covar/{PC}_{i}_cluster_assignation.json",
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
                print(enr.results.head(5))

    # np.corrcoef(xarr, yarr, rowvar=False) #row-wise correlation
    # np.corrcoef(xarr, yarr) #column-wise correlation
