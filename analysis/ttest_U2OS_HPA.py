import sys
sys.path.append('..')
import configs.config as cfg
import pandas as pd
from utils.helpers_hierarchy import factorize_into_quantiles
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from utils.helpers import get_line
from scipy.ndimage import center_of_mass, rotate
from stats_helpers import t_tests_on_groups, plot_example_images, plot_protein_through_umap, resize_with_padding

if __name__ == "__main__":
    df_trans = pd.read_csv(f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}/transformed_matrix.csv")
    n = 3  # Number of quantiles
    pc = 'PC3'
    save_dir = f"{cfg.PROJECT_DIR}/covar/boxplots_{pc}"
    os.makedirs(save_dir, exist_ok=True)
    pc_cells, quantiles, df_trans_tmp = factorize_into_quantiles(df_trans, pc, n)
    df_trans_tmp = df_trans_tmp.rename(columns={'Unnamed: 0': 'image_path'})

    if cfg.CELL_LINE == "S-BIAD34":
        df_trans_tmp['Protein_nu_mean'] = df_trans_tmp['Protein_nu_sum']/df_trans_tmp['nu_area']
        df_trans_tmp['Protein_cyt_mean'] = (df_trans_tmp['Protein_cell_sum']- df_trans_tmp['Protein_nu_sum'])/(df_trans_tmp['cell_area']- df_trans_tmp['nu_area'])
        df_trans_tmp['Protein_cell_mean'] = df_trans_tmp['Protein_cell_sum']/df_trans_tmp['cell_area']
        sc_stats = df_trans_tmp
        mappings = pd.read_csv('/data/HPA-IF-images/IF-image.csv')
        mappings = mappings[mappings.atlas_name=='U2OS']
        ab_loc = mappings[['gene_names','ensembl_ids','antibody','locations']].drop_duplicates()
        sc_stats = sc_stats.merge(ab_loc, right_on='antibody', left_on='ab_id', how='left')
        print(sc_stats.columns[1024:])
        sc_stats["image_path"] = [f"{cfg.PROJECT_DIR}/cell_masks/{ab}/{f}.npy" for ab,f in zip(sc_stats.antibody,sc_stats.cell_id)]
    else:
        try:
            sc_stats = pd.read_csv(f"{cfg.PROJECT_DIR}/cell_nu_ratio.txt")
        except:
            sc_stats = pd.read_csv(f"{cfg.PROJECT_DIR}/single_cell_statistics.csv")
        sc_stats['Protein_nu_mean'] = sc_stats['Protein_nu_sum']/sc_stats['nu_area']
        sc_stats['Protein_cyt_mean'] = (sc_stats['Protein_cell_sum']- sc_stats['Protein_nu_sum'])/(sc_stats['cell_area']- sc_stats['nu_area'])
        sc_stats['Protein_cell_mean'] = sc_stats['Protein_cell_sum']/sc_stats['cell_area']

        mappings = pd.read_csv(f"{cfg.PROJECT_DIR}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border_splitVesiclesPCP.csv")

        if cfg.CELL_LINE == "U2OS":
            df_trans_tmp.image_path = [f.replace('U-2_OS','U2OS') for f in df_trans_tmp.image_path]
            sc_stats = sc_stats.merge(df_trans_tmp[['groups','image_path']], on='image_path', how='left')
            sc_stats = sc_stats.merge(mappings[['cell_idx','sc_target', 'gene_names', 'ensembl_ids']], left_on='image_name', right_on='cell_idx', how='left')
        else:
            df_trans_tmp["cell_id"] = [f.split('/')[-1].replace('.npy','') for f in df_trans_tmp.image_path]
            sc_stats = sc_stats.merge(df_trans_tmp[['groups','cell_id']], on='cell_id', how='left')
            sc_stats = sc_stats.merge(mappings[['cell_idx','sc_target', 'gene_names', 'ensembl_ids']], left_on='cell_id', right_on='cell_idx', how='left')
            sc_stats["image_path"] = [f"{cfg.PROJECT_DIR}/cell_masks/{f}.npy" for f in sc_stats.cell_id]
        print(sc_stats.shape, sc_stats.image_path[:3].values)

        mappings = pd.read_csv('/data/HPA-IF-images/IF-image.csv')
        mappings["image_id"] = [f.split('/')[-1][:-1] for f in mappings.filename]
        try:
            sc_stats["image_id"] = ["_".join(f.split('_')[:-1]) for f in sc_stats.image_name]
        except:
            sc_stats["image_id"] = ["_".join(f.split('_')[:-1]) for f in sc_stats.cell_id]
        sc_stats = sc_stats.merge(mappings[['image_id','antibody','locations']], on='image_id', how='left')


    ############# TODO: Check duplicated cells #############
    #sc_stats = sc_stats.drop_duplicates(subset=['image_name'])
    #print(sc_stats[sc_stats.antibody == 'HPA074371'].groupby('groups').agg({'image_id':'value_counts'}))
    print(sc_stats.groupby('groups').agg({'Protein_nu_mean':'mean',
                                        'Protein_cell_mean':'mean',
                                        'Protein_cyt_mean':'mean'}))

    #boxplots_style1(sc_stats, None, value='Protein_nu_mean', save_dir = save_dir)
    #boxplots_style1(sc_stats, None, value='Protein_cell_mean', save_dir = save_dir)
    #boxplots_style1(sc_stats, None, value='Protein_cyt_mean', save_dir = save_dir)

    if os.path.exists(f"{save_dir}/ttest_{pc}.csv"):
        results = pd.read_csv(f"{save_dir}/ttest_{pc}.csv")
    else:
        results = t_tests_on_groups(sc_stats)
        results.to_csv(f"{save_dir}/ttest_{pc}.csv", index=False)

    results = results[(results.p_1vs2<0.001) | (results.p_2vs3<0.001) | (results.p_1vs3<0.001)]
    results = results[(results.n_group1 >=3) & (results.n_group2 >=3) & (results.n_group3 >=3)]
    print('after fileter results shape: ',results.shape)
    for ab in results.antibody:
        regions = results[results.antibody==ab].region.values
        for value in regions:
            #if value == "Protein_nu_mean":
            #    continue
            boxplots_style1(sc_stats, ab, value=value, save_dir = save_dir)
            plot_example_images(sc_stats, ab, save_dir = save_dir)
    breakme

    # Load inceptionv4 features & labels(cell_id)
    feature_path = "/data/kaggle-dataset/publicHPA_umap/features/d0507_cellv4b_3labels_cls_inception_v3_cbam_i128x128_aug2_5folds/fold0/epoch_12.00_ema/cell_features_test_default_cell_v1.npz"
    features = np.load(feature_path)['feats']
    labels = pd.read_csv("/data/kaggle-dataset/publicHPA_umap/PUBLICHPA/inputs/cells_publicHPA.csv")
    labels["cell_idx"] = [f"{r.ID}_{r.maskid}" for i, r in labels.iterrows()]
    top10_genes = list(results.gene_names)[:10]
    plot_protein_through_umap(features, labels, pc_cell = pc_cells, genes = top10_genes, save_dir = save_dir)


    if False:
        ########### Pathways analysis
        ########### Upregulation
        import gseapy
        databases = [
                "GO_Biological_Process_2023",
                "GO_Cellular_Component_2023",
                "GO_Molecular_Function_2023",
                "WikiPathway_2023_Human",
                "KEGG_2023_Human",
                "CORUM"
            ]
            
        results = pd.read_csv(f"{save_dir}/ttest_U2OS_HPA_{pc}.csv")
        results = results[(results.p_1vs3<0.01) & (results.t_1vs3>1)]
        gene_list = results.gene_names.dropna().unique()
        enr = gseapy.enrichr(gene_list=list(gene_list), gene_sets=databases, organism="human",
                        outdir=f'{save_dir}/enrichr_up', cutoff = 0.5, format='pdf')
        print(enr.results)

        ########### Downregulation
        results = pd.read_csv(f"{save_dir}/ttest_U2OS_HPA_{pc}.csv")
        results = results[(results.p_1vs3<0.01) & (results.t_1vs3<-1)]
        gene_list = results.gene_names.dropna().unique()
        enr = gseapy.enrichr(gene_list=list(gene_list), gene_sets=databases, organism="human",
                        outdir=f'{save_dir}/enrichr_down', cutoff=0.5, format='pdf')
        print(enr.results)
