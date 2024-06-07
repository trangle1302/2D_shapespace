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
import itertools
import glob

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

def find_cells(df_sl, org, n, n_perms, sampled_intensity_dir, intensity_sampling_concentric_ring, intensity_warping):
    ls_ = df_sl[df_sl.sc_target == org].cell_idx.to_list()
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
    return max_n, avg_intensities_perms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_line", help="Cell Line", default="U2OS", type=str)
    args = parser.parse_args()
    intensity_sampling_concentric_ring = False
    intensity_warping = True
    import configs.config as cfg
    cell_line = args.cell_line
    project_dir = os.path.join(os.path.dirname(cfg.PROJECT_DIR), cell_line) 
    save_dir = f"{project_dir}/organelle_stability/pc1_b3_warps"
    os.makedirs(save_dir, exist_ok=True)
    
    shape_mode_path = f"{project_dir}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}"
    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json", "r")
    cells_assigned = json.load(f)
    merged_bins = [[3]]
    n_samples = [1,2,3,5,10,20,50,100,250,500] #[1,2,3,5,10,20]
    n_perms = 100
    PC = 1
    
    fft_dir = f"{project_dir}/fftcoefs/{cfg.ALIGNMENT}"
    fft_path = os.path.join(fft_dir, f"fftcoefs_{cfg.N_COEFS}.txt")
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

    if False:
    # Average organelles
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
                        max_n, avg_intensities_perms = find_cells(df_sl, org, n, n_perms, sampled_intensity_dir, intensity_sampling_concentric_ring, intensity_warping)
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
    if False:
        df = []
        for i, org in enumerate(cfg.ORGANELLES):
            tmp = pd.read_csv(f"{save_dir}/{org}.csv")
            tmp['Organelle'] = org
            df += [tmp]
        df = pd.concat(df)
        print(df.shape, df, df.groupby(['Organelle','sample_n']).agg({'pairwise_correlation':'count'}))
        fig, ax = plt.subplots(nrows=7, ncols=3, sharex=True, sharey=True, figsize=(10, 6))
        for i, org in enumerate(cfg.ORGANELLES):
            print(i//3, i%3)
            df_ = df[df.Organelle==org]
            y = df_['sample_n'].unique()
            X = np.array(df_['pairwise_correlation']).reshape((len(y), -1)).T
            #ax[i//3, i%3] = sns.boxplot(x='sample_n', y='pairwise_correlation', data=df[df.Organelle==org], color="skyblue")
            ax[i//3, i%3].boxplot(x=X, labels = y)
            ax[i//3, i%3].set_title(f'{org}, max_n={df_.max_n.unique()[0]}')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/pc1_b3_concentricrings.png')
        plt.close()

    ### Organelle pairs
    save_dir = f"{project_dir}/organelle_stability/organelle_pairs"
    if False:
        os.makedirs(save_dir, exist_ok=True)
        for (org1, org2) in itertools.combinations(cfg.ORGANELLES,2):
            results = []
            ns = []
            pc_cells = cells_assigned[f"PC{PC}"]   
            for n in [1,2,3,5,10,20,50,100,200]:
                try:
                    for i, bin_ in enumerate(merged_bins):
                        ls = [pc_cells[b] for b in bin_]
                        ls = helpers.flatten_list(ls)
                        ls = [os.path.basename(l).replace(".npy", "") for l in ls]
                        df_sl = mappings[mappings.cell_idx.isin(ls)]
                        max_n1, avg_intensities_perms1 = find_cells(df_sl, org1, n, n_perms, sampled_intensity_dir, intensity_sampling_concentric_ring, intensity_warping)
                        avg_intensities_perms1 = np.array(avg_intensities_perms1)
                        max_n2, avg_intensities_perms2 = find_cells(df_sl, org2, n, n_perms, sampled_intensity_dir, intensity_sampling_concentric_ring, intensity_warping)
                        avg_intensities_perms2 = np.array(avg_intensities_perms2)
                        
                        corr = np.corrcoef(avg_intensities_perms1, avg_intensities_perms2, rowvar=True) #each row represents a variable, with observations in the columns
                        corr = corr[np.triu_indices_from(corr, k=0)] # getting upper triangle of a matrix with the diagonal
                        print(f"{org1}, {org2}, n={n}: {corr.mean()} +/- {corr.std()}")
                        results += corr.tolist()
                        ns += list(np.repeat(n, len(corr)))
                except Exception as e:
                    print(e)
                    continue
            df = pd.DataFrame({'Organelle':f"{org1}-{org2}",
                            'PC': 'PC1',
                            'bin': 'bin3',
                            'max_n': f"{max_n1}-{max_n2}",
                            'max_n1': max_n1,
                            'max_n2': max_n2,
                            'sample_n': ns,
                            'pairwise_correlation': results
                            })
            df.to_csv(f"{save_dir}/{org1}_{org2}.csv", index=False)
            save_path = f'{save_dir}/{org1}-{org2}_boxplot.png'
            average_structure_self_correlation(df, save_path=save_path)

    if True:
        for org in cfg.ORGANELLES:
            df = []
            ls = glob.glob(f"{save_dir}/{org}_*.csv") + glob.glob(f"{save_dir}/*_{org}.csv")
            for l in ls:
                tmp = pd.read_csv(l)
                tmp['Organelle2'] = tmp.Organelle.values[0].replace(org, "").replace("-", "")
                df += [tmp]
            df = pd.concat(df)
            #print(df.shape, df, df.groupby(['Organelle','sample_n']).agg({'pairwise_correlation':'count'}))
            print(org,len(df.Organelle2.unique()))
            fig, ax = plt.subplots(nrows=5, ncols=4, sharex=True, sharey=True, figsize=(10, 8))
            for i, org2 in enumerate(df.Organelle2.unique()):
                print(i//4, i%4)
                df_ = df[df.Organelle2==org2]
                y = df_['sample_n'].unique()
                X = np.array(df_['pairwise_correlation']).reshape((len(y), -1)).T
                ax[i//4, i%4].boxplot(x=X, labels = y)
                ax[i//4, i%4].set_title(f'{org2}, max_n={df_.max_n.unique()[0]}')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/allvs{org}.png')
            plt.close()