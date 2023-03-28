import os
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb
import json 
import numpy as np
from collections import Counter
 
def get_pc_cell_assignment(cells_assigned, PC):
    assignments = dict()
    pc_cells = cells_assigned[PC]
    for b in range(len(pc_cells)):
        cells_ = pc_cells[b]
        for f in cells_:
            assignments.insert({f : b})
    df = pd.DataFrame(assignments)
    print(df)
    return df

def main():    
    project_dir = f"/data/2Dshapespace/S-BIAD34"
    sc_stats = pd.read_csv(f"{project_dir}/single_cell_statistics.csv") 

    alignment = "fft_cell_major_axis_polarized"
    shape_mode_path = f"{project_dir}/shapemode/{alignment}_cell_nuclei_nux4" 
    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json","r")
    cells_assigned = json.load(f)
    print(cells_assigned.keys())

    for PC in cells_assigned.keys():
        df_ = get_pc_cell_assignment(cells_assigned, PC)
        df_.columns()
    breakme
        sc_stats[f"{PC}_bin"] = b

    PC = "PC1"
    pro_count = {}
    for b in np.arange(7):
        pc_cells = cells_assigned[PC][b]
        antibodies = [c.split("/")[-2] for c in pc_cells]
        cells_per_ab = Counter(antibodies)
        pro_count[f"bin{b}"] = cells_per_ab
        print(f"Number of cells in bin{b}: {len(pc_cells)}, Number of proteins: {len(cells_per_ab.keys())}")
    df = pd.DataFrame(pro_count)
    df["total"] = df.sum(axis=1)
    df.sort_values(by=['total'])
    
    # Meta data from the HPA, Antibody 
    ifimages = pd.read_csv("/data/kaggle-dataset/publicHPA_umap/ifimages_U2OS.csv")
    ifimages = ifimages[['ensembl_ids','gene_names','antibody','locations','Ab state']].drop_duplicates()
    ifimages = ifimages[ifimages['Ab state'].isin(['IF_FINISHED','IF_PUBLISHED'])]
    print("Number of ab: ", ifimages.antibody.nunique())

    # meta data
    ab_df = df.iloc[[all(r.values > 5) for _, r in df.iterrows()]]
    ab_df['Antibody id'] = ab_df.index
    meta_df = pd.read_csv(f"{project_dir}/experimentB-processed.txt", sep="\t")
    ab_df = ab_df.merge(meta_df, on=['Antibody id'])
    ab_df = ab_df.merge(ifimages, left_on=['Antibody id'], right_on=["antibody"]) # Only IF_FINISHED here
    print("Number of ab passed QC: ", ab_df.antibody.nunique())

    ab_list = ab_df.antibody.uniques()
    for ab_id in ab_list:
        ab_df_ = sc_stats[sc_stats.an_id == ab_id]
        ab_sc_stats = []
        # plot 
        plt.figure()
        sb.set(style='whitegrid') 
        sb.boxplot(x="timepoint", y="signal", hue="", data=ab_sc_stats)
        plt.savefig(f"{project_dir}/{}")

if __name__ == '__main__':
    main()
