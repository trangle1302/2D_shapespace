import os
from utils.parameterize import get_coordinates
from utils import plotting, helpers, dimreduction, coefs, alignment
from sklearn.decomposition import PCA
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

LABEL_NAMES = {
  0: 'Nucleoplasm',
  1: 'Nuclear membrane',
  2: 'Nucleoli',
  3: 'Nucleoli fibrillar center',
  4: 'Nuclear speckles',
  5: 'Nuclear bodies',
  6: 'Endoplasmic reticulum',
  7: 'Golgi apparatus',
  8: 'Intermediate filaments',
  9: 'Actin filaments',
  10: 'Microtubules',
  11: 'Mitotic spindle',
  12: 'Centrosome',
  13: 'Plasma membrane',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'Vesicles and punctate cytosolic patterns',
  18: 'Negative',
}

all_locations = dict((v, k) for k,v in LABEL_NAMES.items())
#%% Coefficients
fun = "fft"
if fun == "fft":
    get_coef_fun = coefs.fourier_coeffs  # coefs.wavelet_coefs  #
    inverse_func = coefs.inverse_fft  # coefs.inverse_wavelet
elif fun == "wavelet":
    get_coef_fun = coefs.wavelet_coefs
    inverse_func = coefs.inverse_wavelet

if __name__ == "__main__": 
    n_coef = 128
    n_samples = 10000
    n_cv = 10
    cell_line = "U-2 OS" #"S-BIAD34"#"U-2 OS"
    project_dir = "/data/2Dshapespace"
    log_dir = f"{project_dir}/{cell_line.replace(' ','_')}/logs"
    fft_dir = f"{project_dir}/{cell_line.replace(' ','_')}/fftcoefs"
    fft_path = os.path.join(fft_dir,f"fftcoefs_{n_coef}.txt")
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
                    if len(data_[1:]) != n_coef*4:
                        continue
                    #data_dict = {data_dict[0]:data_dict[1:]}
                    lines[data_[0]]=data_[1:]

        df = pd.DataFrame(lines).transpose()
        print(df.shape)
        print(df)
        df = df.applymap(lambda s: np.complex(s.replace('i', 'j'))) 
        shape_mode_path = f"{project_dir}/shapemode/{cell_line}/{i}"
        if not os.path.isdir(shape_mode_path):
            os.makedirs(shape_mode_path)
        
        use_complex = False
        if fun == "fft":
            if not use_complex:
                df_ = pd.concat(
                    [pd.DataFrame(np.matrix(df).real), pd.DataFrame(np.matrix(df).imag)], axis=1
                )
                pca = PCA()
                pca.fit(df_)
                plotting.display_scree_plot(pca, save_dir=shape_mode_path)
            else:
                df_ = df
                pca = dimreduction.ComplexPCA(n_components=df_.shape[1])
                pca.fit(df_)
                plotting.display_scree_plot(pca, save_dir=shape_mode_path)
        elif fun == "wavelet":
            df_ = df
            pca = PCA(n_components=df_.shape[1])
            pca.fit(df_)
            plotting.display_scree_plot(pca, save_dir=shape_mode_path)

        matrix_of_features_transform = pca.transform(df_)
        pc_names = [f"PC{c}" for c in range(1, 1 + len(pca.components_))]
        pc_keep = [f"PC{c}" for c in range(1, 1 + 12)]
        df_trans = pd.DataFrame(data=matrix_of_features_transform.copy())
        df_trans.columns = pc_names
        df_trans.index = df.index
        df_trans[list(set(pc_names) - set(pc_keep))] = 0

        pm = plotting.PlotShapeModes(
            pca,
            df_trans,
            n_coef,
            pc_keep,
            scaler=None,
            complex_type=use_complex,
            inverse_func=inverse_func,
        )
        pm.plot_avg_cell(dark=False, save_dir=shape_mode_path)
        for pc in pc_keep:
            pm.plot_shape_variation_gif(pc, dark=False, save_dir=shape_mode_path)
            pm.plot_pc_dist(pc)
            pm.plot_pc_hist(pc)
            pm.plot_shape_variation(pc, dark=False, save_dir=shape_mode_path)