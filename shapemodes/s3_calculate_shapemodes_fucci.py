import os
import sys
sys.path.append("..")
from shapemodes import dimreduction
from coefficients import coefs
from warps.parameterize import get_coordinates
from utils import plotting, helpers
from sklearn.decomposition import PCA, IncrementalPCA
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import get_location_counts
import glob
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import random
from ast import literal_eval
import json
import resource

LABEL_TO_ALIAS = {
  0: 'Nucleoplasm',
  1: 'NuclearM',
  2: 'Nucleoli',
  3: 'NucleoliFC',
  4: 'NuclearS',
  5: 'NuclearB',
  6: 'EndoplasmicR',
  7: 'GolgiA',
  8: 'IntermediateF',
  9: 'ActinF',
  10: 'Microtubules',
  11: 'MitoticS',
  12: 'Centrosome',
  13: 'PlasmaM',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'VesiclesPCP',
  19: 'Negative',
  19:'Multi-Location',
}

all_locations = dict((v, k) for k,v in LABEL_TO_ALIAS.items())
#%% Coefficients
fun = "fft"
if fun == "fft":
    get_coef_fun = coefs.fourier_coeffs 
    inverse_func = coefs.inverse_fft 
elif fun == "wavelet":
    get_coef_fun = coefs.wavelet_coefs
    inverse_func = coefs.inverse_wavelet
elif fun == "efd":
    get_coef_fun = coefs.elliptical_fourier_coeffs
    inverse_func = coefs.backward_efd

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    #resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def calculate_shapemode(df, n_coef, mode, shape_mode_path=""):
    """ Calculate shapemodes based on the coefficient matrix given
    Parameters
        df: Matrix of coefficient
        n_coef: number of fourier coefficients
        mode: nuclei (shapemode of nuclei) or cell_nuclei (shapemode of both cell and nuclei)
        shape_mode_path: save dir
    Returns
        None
    """
    print(f"Saving to {shape_mode_path}")
    if not os.path.isdir(shape_mode_path):
        os.makedirs(shape_mode_path)
    
    # multiply nucleus by 2:
    print(df.shape)
    print(df.iloc[100,500])
    n_col = df.shape[1]
    df.iloc[:,n_col//2:] = df.iloc[:, n_col//2:].applymap(lambda s: s*4)
    #df = df.applymap(lambda s: s*2, subset = pd.IndexSlice[:,n_col//2 :])
    print(df.iloc[100,500])
    use_complex = False
    if fun == "fft":
        if not use_complex:
            df_ = pd.concat(
                [pd.DataFrame(np.matrix(df).real), pd.DataFrame(np.matrix(df).imag)], axis=1
            )
            pca = PCA() #IncrementalPCA(whiten=True) #PCA()
            pca.fit(df_)
            plotting.display_scree_plot(pca, save_dir=shape_mode_path)
        else:
            df_ = df
            print(f"Number of samples {df_.shape[0]}, number of coefs {df_.shape[1]}")
            pca = dimreduction.ComplexPCA(n_components=df_.shape[1])
            pca.fit(df_)
            plotting.display_scree_plot(pca, save_dir=shape_mode_path)
    elif fun == "wavelet":
        df_ = df
        pca = PCA(n_components=df_.shape[1])
        pca.fit(df_)
        plotting.display_scree_plot(pca, save_dir=shape_mode_path)
    elif fun == "efd":
        df_ = df
        print(f"Number of samples {df_.shape[0]}, number of coefs {df_.shape[1]}")
        pca = PCA(n_components=df_.shape[1])
        pca.fit(df_)
        plotting.display_scree_plot(pca, save_dir=shape_mode_path)
    
    scree = pca.explained_variance_ratio_ * 100
    for percent in np.arange(70,100,5):
        n_pc = np.sum(scree.cumsum() < percent) + 1
        print(f"{n_pc} to explain {percent} % variance")
    n_pc = np.sum(scree.cumsum() < 95) + 1
    n_pc = (8 if n_pc <8 else n_pc) #keep at least 8 PCs or 95% whichever covers the other 
    pc_names = [f"PC{c}" for c in range(1, 1 + len(pca.components_))]
    pc_keep = [f"PC{c}" for c in range(1, 1 + n_pc)]
    matrix_of_features_transform = pca.transform(df_)
    df_trans = pd.DataFrame(data=matrix_of_features_transform.copy())
    df_trans.columns = pc_names
    df_trans.index = df.index
    df_trans[list(set(pc_names) - set(pc_keep))] = 0
    print(matrix_of_features_transform.shape, df_trans.shape)
    
    pm = plotting.PlotShapeModes(
        pca,
        df_trans,
        n_coef,
        pc_keep,
        scaler=None,
        complex_type=use_complex,
        fourier_algo=fun,
        inverse_func=inverse_func,
        mode = mode,
    )
    if mode == "cell_nuclei":
        pm.plot_avg_cell(dark=False, save_dir=shape_mode_path)
    elif mode == "nuclei":
        pm.plot_avg_nucleus(dark=False, save_dir=shape_mode_path)
    
    n_ = 10 # number of random cells to plot
    cells_assigned = dict()
    for pc in pc_keep:
        #pm.plot_shape_variation_gif(pc, dark=False, save_dir=shape_mode_path)
        #pm.plot_pc_dist(pc)
        #pm.plot_pc_hist(pc)
        #pm.plot_shape_variation(pc, dark=False, save_dir=shape_mode_path)
        pc_indexes_assigned, bin_links = pm.assign_cells(pc)
        cells_assigned[pc] = [list(b) for b in bin_links] 
        print(cells_assigned[pc][:3])
    with open(f'{shape_mode_path}/cells_assigned_to_pc_bins.json', 'w') as fp:
        json.dump(cells_assigned, fp)

def main():
    n_coef = 128
    n_samples = -1#5000
    n_cv = 1
    mode = "cell_nuclei"#"nuclei" #"cell_nuclei" #
    cell_line = "S-BIAD34"#"U-2 OS"
    alignment = "fft_cell_major_axis_polarized"#"fft_nuclei_major_axis" #"fft_cell_major_axis_polarized" # 
    project_dir = f"/scratch/users/tle1302/2Dshapespace/{cell_line.replace(' ','_')}" #"/data/2Dshapespace"
    fft_dir = f"{project_dir}/fftcoefs/{alignment}"
    log_dir = f"{project_dir}/logs"
    fft_path = os.path.join(fft_dir, f"fftcoefs_{n_coef}.txt")

    with open(fft_path) as f:
        count = sum(1 for _ in f)
    
    for i in range(n_cv):
        with open(fft_path, "r") as file:
            lines = dict()
            if n_samples ==-1:
                specified_lines = range(count)
            else:
                specified_lines = random.sample(range(count), n_samples) # 10k cells/ CV
            # loop over lines in a file
            for pos, l_num in enumerate(file):
                # check if the line number is specified in the lines to read array
                #print(pos)
                if pos in specified_lines:
                    # print the required line number
                    data_ = l_num.strip().split(',')
                    if fun == "efd":
                        if len(data_[1:]) != (n_coef*4+2)*2:
                            continue
                    elif len(data_[1:]) != n_coef*4:
                            continue
                    #data_dict = {data_dict[0]:data_dict[1:]}
                    lines[data_[0]]=data_[1:]

        df = pd.DataFrame(lines).transpose()
        if fun == "fft":
            df = df.applymap(lambda s: complex(s.replace('i', 'j'))) 
        
        if mode == "nuclei":
            df = df.iloc[:,(df.shape[1]//2):]
        elif mode == "cell":
            df = df.iloc[:,:(df.shape[1]//2)]
        print(cell_line, alignment, mode, df.shape)

        shape_mode_path = f"{project_dir}/shapemode/{alignment}_{mode}_nux4"
        calculate_shapemode(df, n_coef, mode, shape_mode_path=shape_mode_path)
        
        # Shape modes of G1, G2/S, G2 cells:
        sc_stats = pd.read_csv(f"{project_dir}/single_cell_statistics.csv", index=False)
        for cells_in_phase in sc_stats.groupby("GMM"):
            print(f"Index format: {df.index[:4]}")
            df_ = df[df.index.isin(cells_in_phase.cell_id)]
            save_dir = f"{project_dir}/shapemode/{alignment}_{mode}_{cells_in_phase.GMM.values[0]}"
            calculate_shapemode(df_, n_coef, mode, shape_mode_path=save_dir)

if __name__ == "__main__": 
    memory_limit() # Limitates maximun memory usage
    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)
