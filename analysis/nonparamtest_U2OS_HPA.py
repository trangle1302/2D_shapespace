import sys
sys.path.append('..')
import configs.config as cfg
import pandas as pd
from utils.helpers_hierarchy import factorize_into_quantiles
from scipy.stats import ttest_ind, kruskal, false_discovery_control
import os

from stats_helpers import kruskal_wallis_test, boxplots_style2, plot_example_images


if __name__ == "__main__":
    
    df_trans = pd.read_csv(f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}/transformed_matrix.csv")
    n = 3  # Number of quantiles
    for pc in ['PC1','PC2','PC3','PC4','PC5','PC6']:
        save_dir = f"{cfg.PROJECT_DIR}/kruskal/boxplots_{pc}"
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

        #print(sc_stats.columns[1024:], sc_stats.shape)
        ############# Check duplicated cells #############
        sc_stats = sc_stats.drop_duplicates(subset=['image_path'])
        #print(sc_stats.shape)
        #print(sc_stats[sc_stats.antibody == 'HPA074371'].groupby('groups').agg({'image_id':'value_counts'}))
        print(sc_stats.groupby('groups').agg({'Protein_nu_mean':'mean',
                                            'Protein_cell_mean':'mean',
                                            'Protein_cyt_mean':'mean'}))

        #boxplots_style1(sc_stats, None, value='Protein_nu_mean', save_dir = save_dir)
        #boxplots_style1(sc_stats, None, value='Protein_cell_mean', save_dir = save_dir)
        #boxplots_style1(sc_stats, None, value='Protein_cyt_mean', save_dir = save_dir)

        plot_example_images(sc_stats, "HPA055941", save_dir = save_dir)
        breakme
        if os.path.exists(f"{save_dir}/kruskal_{pc}.csv"):
            results = pd.read_csv(f"{save_dir}/kruskal_{pc}.csv")
        else:
            results = kruskal_wallis_test(sc_stats)
            results.to_csv(f"{save_dir}/kruskal_{pc}.csv", index=False)
        print('results shape: ', results.shape)
        print(results)
        group_top10 = results.sort_values(by='p', ascending=True).iloc[:10,:]
        for i, r in group_top10.iterrows():
            boxplots_style2(sc_stats, r.antibody, value = r.region, save_dir = save_dir)
            plot_example_images(sc_stats, r.antibody, save_dir = save_dir)
            print(r.antibody, r.gene_names, r.location, r.p)
            print('-----------------------------------')

    # Combine results and FDR
    import glob
    tests = glob.glob(f"{cfg.PROJECT_DIR}/kruskal/boxplots_*/kruskal_*")
    combined_results = []
    for test in tests:
        results = pd.read_csv(test)
        results['PC'] = test.split('/')[-2].split('_')[-1]
        combined_results.append(results)
    combined_results = pd.concat(combined_results)
    combined_results['FDR_BH'] = false_discovery_control(combined_results.p, method='BH') # Benjamini-Hochberg
    combined_results['FDR_BY'] = false_discovery_control(combined_results.p, method='BY') # Benjamini-Yekutieli, much stricter
    print(combined_results[combined_results.FDR_BH < 0.05].groupby('PC').agg({'antibody':'count'}))
    print(combined_results[combined_results.FDR_BY < 0.05].groupby('PC').agg({'antibody':'count'}))
    combined_results.to_csv(f"{cfg.PROJECT_DIR}/kruskal/combined_kruskal.csv", index=False)
    tmp = combined_results[combined_results.FDR_BH < 0.05].groupby('PC')
    for pc, group in tmp:
        print(pc, group[['antibody','gene_names','location','p','FDR_BH']])
        glsig = group.gene_names.unique()
        group.to_csv(f"{cfg.PROJECT_DIR}/kruskal/{pc}_genelist_sig_kruskal.csv", index=False)
        print(pc, len(glsig))
        with open(f"{cfg.PROJECT_DIR}/kruskal/{pc}_genelist_sig_kruskal.txt", 'w') as outfile:
            outfile.write('\n'.join(str(i) for i in glsig))
