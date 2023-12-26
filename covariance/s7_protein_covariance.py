import os
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import argparse
from utils import helpers
import glob
from skimage.filters import threshold_otsu
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--PC", help="shape mode", type=str)
    args = parser.parse_args()
    print(args.PC)
    import configs.config as cfg
    
    log_dir = f"{cfg.PROJECT_DIR}/logs"
    fftcoefs_dir = f"{cfg.PROJECT_DIR}/fftcoefs"
    fft_path = os.path.join(fftcoefs_dir, f"fftcoefs_{cfg.N_COEFS}.txt")
    shape_mode_path = f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}"
    sampled_intensity_dir = f"{cfg.PROJECT_DIR}/sampled_intensity_bin"

    mappings = pd.read_csv(f"{cfg.PROJECT_DIR}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border_splitVesiclesPCP.csv")
    id_with_intensity = glob.glob(f"{sampled_intensity_dir}/*.npy")
    print(mappings, len(id_with_intensity))
    mappings["Link"] = [
        f"{sampled_intensity_dir}/{id.split('_',1)[1]}_protein.npy"
        for id in mappings.id
    ]
    mappings = mappings[mappings.Link.isin(id_with_intensity)]
    print(mappings.target.value_counts(), mappings)

    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json")
    cells_assigned = json.load(f)
    print(cells_assigned.keys())
    save_dir = f"{cfg.PROJECT_DIR}/covar"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    meta = []
    #merged_bins = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
    merged_bins = [[0, 1, 2], [3], [4, 5, 6]]
    PC = args.PC
    pc_cells = cells_assigned[f"PC{PC}"]
    if True:  # for PC, pc_cells in cells_assigned.items():
        shape = (21, cfg.N_COEFS * 2)
        intensities_pcX = []
        counts = []
        for i, bin_ in enumerate(merged_bins):
            if os.path.exists(f"{save_dir}/PC{PC}_{i}_intensities.csv"):
                intensities = pd.read_csv(f"{save_dir}/PC{PC}_{i}_intensities.csv")
            else:
                intensities = []
                ensembl_ids = []
                ls = [pc_cells[b] for b in bin_]
                ls = helpers.flatten_list(ls)
                ls = [
                    f"{sampled_intensity_dir}/{os.path.basename(l)}".replace(
                        ".npy", "_protein.npy"
                    )
                    for l in ls
                ]
                print(ls[:3])
                print(mappings.Link)
                df_sl = mappings[mappings.Link.isin(ls)]
                print(
                    f"PC{PC}: Number of cells {df_sl.shape[0]}, Number of gene: {df_sl.ensembl_ids.nunique()}"
                )
                for _, row in df_sl.iterrows():
                    intensity = np.load(row.Link)
                    thres = threshold_otsu(intensity)
                    intensity = np.where(intensity > thres, 1, 0)
                    intensities += [intensity.flatten()]
                    ensembl_ids += [row.ensembl_ids]
                intensities = pd.DataFrame(intensities)
                intensities["ensembl_ids"] = ensembl_ids
                intensities = intensities.groupby("ensembl_ids").agg("mean")
                intensities.to_csv(f"{save_dir}/PC{PC}_{i}_intensities.csv")
                """
                #intensities_pcX += [np.nanmean(intensities, axis=0).reshape(intensity.shape)]
                covar_mat = np.corrcoef(np.array(intensities.drop(["ensembl_ids"], axis=1)))
                covar_mat = pd.DataFrame(covar_mat)
                covar_mat.column = ensembl_ids
                covar_mat.index = ensembl_ids
                covar_mat.to_csv(f"{save_dir}/{}.csv")
                """

                intensities.index = intensities.ensembl_ids
                intensities = intensities.drop("ensembl_ids", axis=1)
                intensities = intensities.transpose().astype("float32")
                intensities.to_csv(f"{save_dir}/PC{PC}_{i}_intensities.csv")

            """
            if os.path.exists(f"{save_dir}/PC{PC}_{i}.csv"):
                covar_mat = pd.read_csv(f"{save_dir}/PC{PC}_{i}.csv")
            else:
                covar_mat = intensities.corr()
                covar_mat = covar_mat.astype("float32") 
                covar_mat.to_csv(f"{save_dir}/PC{PC}_{i}.csv")
            print(covar_mat.shape)
            print(covar_mat.columns[:3])
            
            if "ensembl_ids" in covar_mat.column:
                corr = covar_mat.drop("ensembl_ids", axis=1)#.values
            else:
                corr = covar_mat
            print(corr.shape)
            
            pdist = spc.distance.pdist(corr)
            if pdist.dtype == "float64":
                pdist = pdist.astype("float32")
            linkage = spc.linkage(pdist, method='complete')
            idx = spc.fcluster(linkage, 0.3 * pdist.max(), 'distance')
            cluster_assignation = {"assignation": [int(i) for i in idx],
                                    "ensembl_ids": covar_mat.columns.tolist(),
                                    "max_intensity": intensities.max(axis=0).tolist(),
                                    "mean_intensity": intensities.mean(axis=0).tolist()}
            with open(f"{save_dir}/PC{PC}_{i}_cluster_assignation.json", "w") as fp:
                json.dump(cluster_assignation, fp)
            
            for ii in np.unique(idx):
                p = sns.heatmap(covar_mat.iloc[np.where(idx==ii)[0],np.where(idx==ii)[0]], 
                            cmap='RdBu', vmin=-1, vmax=1)
                p.get_figure().savefig(f"{save_dir}/PC{PC}_bin{i}_cluster{ii}.png", bbox_inches="tight")
                plt.close()
            
            # Plot
            p = sns.clustermap(corr, method="complete", cmap='RdBu', annot=True, 
               annot_kws={"size": 3}, vmin=-1, vmax=1, figsize=(20,20))
            p.savefig(f"{save_dir}/PC{PC}_{i}.png")
            # covar matrix is symmetric, so getting row dendogram is the same as col dendogram
            dendogram = p.dendrogram_row.dendrogram
            Z = p.dendrogram_col.linkage
            max_d = 0.3 * np.max(spc.distance.pdist(p.dendrogram_col.array))
            clusters = spc.fcluster(Z, max_d, criterion='distance') # original order!
            dendogram["clusters"] = clusters.tolist()  #" ".join([str(elem) for elem in clusters]) 
            dendogram["clusters_reordered"] = list(clusters.astype('int')[p.dendrogram_col.reordered_ind])
            with open(f"{save_dir}/PC{PC}_{i}_cluster_assignation2.json", "w", encoding='utf-8') as fp:
                json.dump(dendogram, fp, cls=npEncoder)
            """
            """
    #np.corrcoef(xarr, yarr, rowvar=False) #row-wise correlation
    #np.corrcoef(xarr, yarr, rowvar=False) #column-wise correlation
    """
