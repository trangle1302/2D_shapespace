import os
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import argparse
from utils import helpers
import matplotlib.pyplot as plt
import json
import scipy.cluster.hierarchy as spc
import seaborn as sns
from skimage.filters import threshold_otsu
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--PC", help="shape mode", type=str)
    args = parser.parse_args()
    print(args.PC)
    import configs.config as cfg
    
    """
    #project_dir = f"/scratch/users/tle1302/2Dshapespace/{cell_line.replace(' ','_')}"
    project_dir = "/data/2Dshapespace/U-2_OS"
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
    """
    sampled_intensity_dir = f"{cfg.PROJECT_DIR}/sampled_intensity_bin"
    mappings = pd.read_csv(f"{cfg.PROJECT_DIR}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border_splitVesiclesPCP.csv")
    mappings["Link"] = [sampled_intensity_dir + "/"+ f.split("_",1)[1] + "_protein.npy" for f in mappings.id]
    
    f = open(f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}/cells_assigned_to_pc_bins.json")
    cells_assigned = json.load(f)
    
    save_dir = f"{cfg.PROJECT_DIR}/covar"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    meta = []
    #merged_bins = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
    merged_bins = [[0, 1, 2], [3], [4, 5, 6]]
    PC = args.PC
    pc_cells = cells_assigned[f"PC{PC}"]
    if True:  # for PC, pc_cells in cells_assigned.items():
        s = time.time()
        shape = (21, cfg.N_COEFS * 2)
        intensities_pcX = []
        counts = []
        for i, bin_ in enumerate(merged_bins):
            if i == 0:
                continue
            if os.path.exists(f"{save_dir}/PC{PC}_{i}.csv"):
                covar_mat = pd.read_csv(f"{save_dir}/PC{PC}_{i}.csv")
            else:
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
                    print(mappings.Link[0])
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
                if "ensembl_ids" in intensities.columns:
                    intensities.index = intensities.ensembl_ids
                    intensities = intensities.drop("ensembl_ids", axis=1)
                    intensities = intensities.transpose().astype("float32")
                    intensities.to_csv(f"{save_dir}/PC{PC}_{i}_intensities.csv")            
                covar_mat = intensities.corr()
                covar_mat = covar_mat.astype("float32")
                covar_mat.to_csv(f"{save_dir}/PC{PC}_{i}.csv")
            
            covar_mat = covar_mat[covar_mat.columns.drop(list(covar_mat.filter(regex='Unnamed:')))]
            corr = covar_mat.values
            pdist = spc.distance.pdist(corr)
            if pdist.dtype == "float64":
                pdist = pdist.astype("float32")
            linkage = spc.linkage(pdist, method="ward")
            print(f"Assigning clusters by distance threshold { 0.3 * pdist.max()}, max distance {pdist.max()}")
            idx = spc.fcluster(linkage, 0.3 * pdist.max(), "distance")
            cluster_assignation = {
                "assignation": [int(i) for i in idx],
                "ensembl_ids": covar_mat.columns.tolist(),
            }
            with open(f"{save_dir}/PC{PC}_{i}_cluster_assignation.json", "w") as fp:
                json.dump(cluster_assignation, fp)

            for ii in np.unique(idx):
                tmp=covar_mat.iloc[np.where(idx == ii)[0], np.where(idx == ii)[0]]
                #print(tmp.shape, tmp)
                print(covar_mat.columns[np.where(idx == ii)[0]])
                print(covar_mat.max(),covar_mat.min(),covar_mat.mean())
                plt.figure()
                p = sns.heatmap(
                    tmp,
                    cmap="RdBu",
                    vmin=covar_mat.min(),
                    vmax=covar_mat.max(),
                )
                plt.savefig(
                    f"{save_dir}/PC{PC}_bin{i}_cluster{ii}.png", bbox_inches="tight"
                )
                plt.close()
            if False:
                # Plot
                plt.figure()
                p = sns.clustermap(covar_mat, method="ward", cmap='RdBu', annot=True, 
                annot_kws={"size": 3}, vmin=covar_mat.min(), vmax=covar_mat.max(), figsize=(20,20))
                plt.setp(p.get_xticklabels(), rotation=45, horizontalalignment='right')
                p.savefig(f"{save_dir}/PC{PC}_{i}.png", bbox_inches="tight")
                plt.close()
            
            """
            # covar matrix is symmetric, so getting row dendogram is the same as col dendogram
            dendogram = p.dendrogram_row.dendrogram
            Z = p.dendrogram_col.linkage
            max_d = 0.3 * np.max(spc.distance.pdist(p.dendrogram_col.array))
            clusters = spc.fcluster(Z, max_d, criterion='distance') # original order!
            dendogram["clusters"] = clusters.tolist()  #" ".join([str(elem) for elem in clusters]) 
            dendogram["clusters_reordered"] = list(clusters.astype('int')[p.dendrogram_col.reordered_ind])
            with open(f"{save_dir}/PC{PC}_{i}_cluster_assignation.json", "w", encoding='utf-8') as fp:
                json.dump(dendogram, fp, cls=npEncoder)
            """
        print(f"Time elapsed: {(time.time() - s)/3600} h")
    # np.corrcoef(xarr, yarr, rowvar=False) #row-wise correlation
    # np.corrcoef(xarr, yarr, rowvar=False) #column-wise correlation
