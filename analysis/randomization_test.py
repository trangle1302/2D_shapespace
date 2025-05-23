import os
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import configs.config as cfg
from stats_helpers import *

PERMUTATIONS = 10000

def get_r2_numpy_corrcoef(x, y):
    return np.corrcoef(x, y)[0, 1]**2

def mvavg(yvals, mv_window):
    '''Calculate the moving average'''
    return np.convolve(yvals, np.ones((mv_window,))/mv_window, mode='valid')

def log_min_max_norm(x):
    x = np.log10(x) + 0.000001
    return (x-np.min(x))/(np.max(x)-np.min(x))

def max_norm(x):
    return x/np.max(x)

def permutation_analysis(transformed_matrix, PCs=['PC1','PC2','PC3']):
    '''Permutation analysis for the moving average of the PCs
    Parameters
        transformed_matrix: pandas dataframe with transformed data, contains columns 'ab_id', 'Protein_nu_mean', 'Protein_cyt_mean', 'MT_cell_mean', 'pseudotime'
        PCs: list of PCs to analyze
    Returns 
        pandas dataframe of the results
    '''
    feature1 = 'Protein_nu_mean'
    feature2 = 'Protein_cyt_mean'
    mv_window = 20
    results = []
    for PC in PCs: #range(1,11):
        for ab_, df_ in transformed_matrix.groupby('ab_id'):
            sorted_indices = np.argsort(df_[PC])
            # Sort data along PCx
            sorted_feature1 = df_[feature1].values[sorted_indices.values]
            sorted_feature2 = df_[feature2].values[sorted_indices.values]
            #sorted_pos = df_[PC].values[sorted_indices.values]
            sorted_mt = df_['MT_cell_mean'].values[sorted_indices.values]
            #sorted_speudotime = df_['pseudotime'].values[sorted_indices.values]

            # Remove outliers:
            values = sorted_indices.copy() 
            sorted_indices = remove_outliers(values, sorted_indices)
            sorted_feature1 = remove_outliers(values, sorted_feature1)
            sorted_feature2 = remove_outliers(values, sorted_feature2)
            sorted_mt = remove_outliers(values, sorted_mt)

            # Normalize/standardize
            sorted_feature1 = max_norm(sorted_feature1)
            sorted_feature2 = max_norm(sorted_feature2)
            sorted_mt = max_norm(sorted_mt)

            # Apply moving average
            #sorted_mt_mvavg = mvavg(sorted_mt, mv_window)
            sorted_feature1_mvavg = mvavg(sorted_feature1, mv_window)
            sorted_feature2_mvavg = mvavg(sorted_feature2, mv_window)
            #sorted_pc_mvavg = mvavg(sorted_pos, mv_window)

            # Permutations
            perms = [np.random.permutation(len(df_[feature1].values)) for _ in range(PERMUTATIONS)]
            features = max_norm(df_[feature1].values)
            # Metric : mean difference from random
            curr_rng_comp = [features[perm] for perm in perms]
            curr_mvavg_rng_comp = [mvavg(rng_feats, mv_window) for rng_feats in curr_rng_comp]
            pervar_ = np.var(sorted_feature1_mvavg)/np.var(features)
            pervar_ordered = np.var(curr_mvavg_rng_comp,axis=1) / np.var(features)
            mean_diff_1 = np.mean(pervar_ - pervar_ordered)


            # Permutations
            perms = [np.random.permutation(len(df_[feature2].values)) for _ in range(PERMUTATIONS)]
            features = max_norm(df_[feature2].values)
            # Metric : mean difference from random
            curr_rng_comp = [features[perm] for perm in perms]
            curr_mvavg_rng_comp = [mvavg(rng_feats, mv_window) for rng_feats in curr_rng_comp]
            pervar_ = np.var(sorted_feature2_mvavg)/np.var(features)
            pervar_ordered = np.var(curr_mvavg_rng_comp,axis=1) / np.var(features)
            mean_diff_2 = np.mean(pervar_ - pervar_ordered)
            results += [[ab_,PC, mean_diff_1, mean_diff_2]]
            #results[ab_] = {PC: {'mean_diff_nu': mean_diff_1, 'mean_diff_cyt': mean_diff_2}}
    results = pd.DataFrame(results)
    results.columns = ['ab','PC','mean_diff_nu','mean_diff_cyt']
    return results

def mvavg_ci(yvals, window_size, ci = 0.95):
    '''Calculate the moving confidence interval    
    '''
    min_max = []
    for i in range(len(yvals) - window_size + 1):
        a = yvals[i: i + window_size]
        min_max += [st.t.interval(ci, len(a)-1, loc=np.mean(a), scale=st.sem(a))]
    return np.array(min_max)

def remove_outliers_idx(values, n_std=5):
    '''Returns indices of outliers to keep'''
    max_cutoff = np.mean(values) + n_std * np.std(values)
    min_cutoff = np.mean(values) - n_std * np.std(values)
    return (values < max_cutoff) & (values > min_cutoff)
    
def remove_outliers(values, return_values):
    '''Remove outliers on "values" and return "return_values" based on that filter'''
    return return_values[remove_outliers_idx(values)]

def plot_moving_averages(df,ab, PC, feature_name, mv_window = 20, rm_outliers=True):
    df_ = df[df.ab_id==ab]
    # print(f'Number of cells: {df_.shape[0]}, {df_[feature_name].describe()}')
    df_ = df_[remove_outliers_idx(df_[feature_name],5)]
    #df_ = df_[remove_outliers_idx(df_.nu_area,2)]
    # print(f'Number of cells after filter 5std from mean intensity: {df_.shape[0]}')
    
    # remove 'bad' data point: when nucleus is larger than the cell
    #df_ = df_[df_[feature_name]>0]
    sorted_indices = np.argsort(df_[PC])
    # Sort data along PCx
    sorted_feature1 = df_[feature_name].values[sorted_indices.values]
    sorted_pos = df_[PC].values[sorted_indices.values]    
    sorted_mt = df_['MT_cell_mean'].values[sorted_indices.values]
    sorted_speudotime = df_['pseudotime'].values[sorted_indices.values]
    
    if rm_outliers:
        # remove outliers position
        values = sorted_pos.copy() #
        #values = sorted_feature1.copy()
        sorted_pos = remove_outliers(values, sorted_pos)
        sorted_mt = remove_outliers(values, sorted_mt)
        sorted_speudotime = remove_outliers(values, sorted_speudotime)
        sorted_feature1 = remove_outliers(values, sorted_feature1)

    # print(f'Number of cells after filter 5std from mean position: {df_.shape[0]}')
    
    # Normalize/standardize
    sorted_feature1 = max_norm(sorted_feature1) # log_min_max_norm(sorted_feature1)
    sorted_mt = max_norm(sorted_mt) #log_min_max_norm(sorted_mt)
    
    # Apply moving average
    sorted_mt_mvavg = mvavg(sorted_mt, mv_window)
    sorted_feature1_mvavg = mvavg(sorted_feature1, mv_window)
    sorted_pc_mvavg = mvavg(sorted_pos, mv_window)
    
    # Plots
    plt.figure()
    plt.scatter(sorted_pos, sorted_feature1, color='blue', alpha=0.1, label='protein intensity')
    plt.plot(sorted_pc_mvavg, sorted_feature1_mvavg, c='blue')
    # plt.scatter(sorted_pos, sorted_mt, color='grey', alpha=0.1, label='MT intensity')
    # plt.plot(sorted_pc_mvavg, sorted_mt_mvavg, c='grey')
    #plt.plot(sorted_pc_mvavg, sorted_feature2_mvavg, c='red')
    #mvavg_min = pd.Series(sorted_feature1).rolling(mv_window).min()[mv_window-1:]
    #mvavg_max = pd.Series(sorted_feature1).rolling(mv_window).max()[mv_window-1:]
    ci_minmax = mvavg_ci(sorted_feature1, mv_window, ci = 0.95)
    plt.vlines(sorted_pc_mvavg, ci_minmax[:,0], ci_minmax[:,1], color="blue", alpha=0.2)
    plt.title(f'{ab}: {PC} vs {feature_name}')
    plt.legend()

if __name__ == "__main__":

    df = pd.read_csv(f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}/transformed_matrix.csv")
    df = df.drop('Unnamed: 0', axis=1)
    df['Protein_nu_mean'] = df['Protein_nu_sum']/df['nu_area']
    df['Protein_cyt_mean'] = (df['Protein_cell_sum']- df['Protein_nu_sum'])/(df['cell_area']- df['nu_area'])
    df['Protein_cell_mean'] = df['Protein_cell_sum']/df['cell_area']
    df['CDT1_nu_mean'] = df['CDT1_nu_sum']/df['nu_area']
    df['MT_cell_mean'] = df['MT_cell_sum']/df['cell_area']
    #mappings = pd.read_csv(f"/mnt/c/Users/trang.le/Desktop/shapemode/S-BIAD34/experimentB-processed.txt", sep="\t")
    mappings = pd.read_csv(f"{cfg.PROJECT_DIR}/CellCycleVariationSummary.csv")
    ifimages = pd.read_csv(f"/data/HPA-IF-images/IF-image.csv")
    ifimages = ifimages[ifimages.atlas_name=='U2OS']
    ifimages = ifimages[ifimages.latest_version==23]
    ab_loc = ifimages[['antibody','locations','gene_names']].drop_duplicates()
    mappings = mappings.merge(ab_loc, left_on='antibody', right_on='antibody')
    print(mappings.columns)
    hit_path = f"{cfg.PROJECT_DIR}/protein_expression_permutation_through_shapes.csv"
    if not os.path.exists(hit_path):
        results_pcs = permutation_analysis(df, PCs=['PC1','PC2','PC3','PC4','PC5', 'PC6'])
        results_pcs.to_csv(hit_path, index=False)
    else:
        results_pcs = pd.read_csv(hit_path)

    try:
        results_pcs = results_pcs.drop('Unnamed: 0', axis=1).drop_duplicates()
    except:
        results_pcs = results_pcs.drop_duplicates()
    results_pcs = results_pcs.merge(mappings[['antibody','gene_names','locations','ccd_reason']], left_on='ab', right_on='antibody')
    results_pcs.to_csv(hit_path.replace('.csv','ccd.csv'), index=False)
    # Visualization
    save_dir = f"{cfg.PROJECT_DIR}/randomization_test"
    os.makedirs(save_dir, exist_ok=True)
    top20_nu = results_pcs[results_pcs.locations.fillna('').str.contains('Nuc')] 
    top20_nu = top20_nu.sort_values('mean_diff_nu', ascending=False).iloc[:20,:]
    top20_nu.to_csv(f"{save_dir}/top20_nu.csv", index=False)
    print(top20_nu)
    for i, r in top20_nu.iterrows():
        genename = mappings[mappings.antibody == r.ab].gene_names.values
        plt.figure()
        plot_moving_averages(df,r.ab,r.PC,'Protein_nu_mean', mv_window = 20, rm_outliers=True)
        plt.savefig(f'{save_dir}/{r.PC}_nu_{r.ab}_{genename}.png')

        
        #plt.figure()
        #plot_moving_averages(df,r.ab,'pseudotime','Protein_nu_mean', mv_window = 20, rm_outliers=True)
        #plt.savefig(f'{save_dir}/pseudotime_nu_{r.ab}_{genename}.png')
        #breakme

    top20_cyt = results_pcs[~results_pcs.fillna('').locations.str.contains('Nuc')] 
    top20_cyt = top20_cyt.sort_values('mean_diff_cyt', ascending=False).iloc[:20,:]
    top20_cyt.to_csv(f"{save_dir}/top20_cyt.csv", index=False)
    print(top20_cyt)
    for i, r in top20_cyt.iterrows():
        genename = mappings[mappings.antibody == r.ab].gene_names.values
        plt.figure()
        plot_moving_averages(df,r.ab,r.PC,'Protein_cyt_mean', mv_window = 20, rm_outliers=True)
        plt.savefig(f'{save_dir}/{r.PC}_cyt_{r.ab}_{genename}.png')
