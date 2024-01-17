import sys
sys.path.append('../')
import os
import json
from utils import helpers
import numpy as np 
import pandas as pd
from imageio import imread, imwrite
import argparse
from organelle_heatmap import unmerge_label, get_average_intensities_cr, get_average_intensities_tsp
from organelle_heatmap_cell_pairwise import load_intensities
import matplotlib.pyplot as plt
import seaborn as sns

def average_structure_self_correlation(df, save_path):
    # df should have these columns ['Organelle', 'sample_n','max_n', 'pairwise_correlation']
    # Create a boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='sample_n', y='pairwise_correlation', data=df, color="skyblue")

    # Add individual data points
    #sns.stripplot(x='sample_n', y='pairwise_correlation', data=df, color=".25", alpha=0.05)

    # X axis labels 
    plt.xlabel(f"{df.Organelle.unique()[0]}, max_n={df.max_n.unique()[0]}")
    #plt.ylabel(f"Pairwise correlation")
    labels = [f"n={n}" for n in df.sample_n.unique()]
    print(df.sample_n.unique(), labels)
    plt.xticks(list(range(len(labels))), labels)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_line", help="Cell Line", type=str)
    args = parser.parse_args()
    intensity_sampling_concentric_ring = False
    intensity_warping = True

    import configs.config as cfg
    cell_line = args.cell_line
    project_dir = os.path.join(os.path.dirname(cfg.PROJECT_DIR), cell_line) 
    fft_dir = f"{project_dir}/fftcoefs/{cfg.ALIGNMENT}"
    fft_path = os.path.join(fft_dir, f"fftcoefs_{cfg.N_COEFS}.txt")
    shape_mode_path = f"{project_dir}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}"
    if intensity_sampling_concentric_ring:
        sampled_intensity_dir = f"{project_dir}/sampled_intensity_bin"
    if intensity_warping:
        sampled_intensity_dir = f"{project_dir}/warps" 
    
    cellline_meta = os.path.join(project_dir, os.path.basename(cfg.META_PATH).replace(".csv", "_splitVesiclesPCP.csv"))
    if os.path.exists(cellline_meta):
        mappings = pd.read_csv(cellline_meta)
    else:
        mappings = pd.read_csv(cfg.META_PATH)
        mappings = mappings[mappings.atlas_name == cell_line]
        mappings["cell_idx"] = [idx.split("_", 1)[1] for idx in mappings.id]
        mappings = unmerge_label(mappings)
        mappings.to_csv(cellline_meta, index=False)
        print(mappings.sc_target.value_counts())
    #print(mappings.sc_target.value_counts())
    mappings.loc[mappings.sc_target=="Cytoplasmic bodies","sc_target"]="CytoBodies"
    mappings.loc[mappings.sc_target=="Lipid droplets","sc_target"]="LipidDrop"
    print(mappings.columns, mappings.sc_target.value_counts())
    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json", "r")
    cells_assigned = json.load(f)
    merged_bins = [[0,1,2,3,4,5,6]]
    
    save_dir = f"{project_dir}/organelle_stability/all"
    os.makedirs(save_dir, exist_ok=True)
    # Average organelles
    n_samples = [1,2,3,5,10,20,50,100,250,500] #[1,2,3,5,10,20]
    n_perms = 100
    PC = 1
    for org in cfg.ORGANELLES:
        results = []
        ns = []
        pc_cells = cells_assigned[f"PC{PC}"]   
        for n in n_samples:
            try:
                for i, bin_ in enumerate(merged_bins):
                    ls = [pc_cells[b] for b in bin_]
                    ls = helpers.flatten_list(ls)
                    ls = [os.path.basename(l).replace(".npy", "") for l in ls]
                    df_sl = mappings[mappings.cell_idx.isin(ls)]
                    ls_ = df_sl[df_sl.sc_target == org].cell_idx.to_list()
                    if len(ls_)==0:
                        print(f"{org}: Found {len(ls_)}, eg: {ls_[:3]}")
                        continue
                    max_n = len(ls_)

                    avg_intensities_perms = []
                    for i in range(n_perms):
                        ls_cur = np.random.choice(ls_, n, replace=False)
                        if intensity_sampling_concentric_ring:
                            intensities = get_average_intensities_cr(ls_cur, sampled_intensity_dir=sampled_intensity_dir)
                        if intensity_warping:
                            intensities = get_average_intensities_tsp(ls_cur, sampled_intensity_dir=sampled_intensity_dir)
                            intensities = (intensities*255).astype('uint8')
                        avg_intensities_perms.append(intensities.flatten())
                    avg_intensities_perms = np.array(avg_intensities_perms)
                    corr = np.corrcoef(avg_intensities_perms) 
                    corr = corr[np.triu_indices_from(corr, k=1)] # getting upper triangle of a matrix without the diagonal (corrcoef==1)
                    print(f"{org}, n={n}: {corr.mean()} +/- {corr.std()}")
                    results += corr.tolist()
                    ns += list(np.repeat(n, len(corr)))
            except Exception as e:
                print(e)
                continue
            df = pd.DataFrame({'Organelle':org,
                            'PC': 'PC1',
                            'bin': 'bin3',
                            'max_n': max_n,
                            'sample_n': ns,
                            'pairwise_correlation': results
                            })
        df.to_csv(f"{save_dir}/{org}.csv", index=False)
        save_path = f'{save_dir}/{org}_boxplot.png'
        average_structure_self_correlation(df, save_path=save_path)

    