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
import pickle
import time

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

def calculate_fft_hpa():
    cell_line = "U-2 OS"
    save_dir = f"/data/2Dshapespace/{cell_line.replace(' ','_')}/cell_masks"
    d = Path(save_dir)
    save_path = Path(f"/data/2Dshapespace/{cell_line.replace(' ','_')}/fftcoefs")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_dir = f"/data/2Dshapespace/{cell_line.replace(' ','_')}/logs"

    imlist= glob.glob(f"{save_dir}/*.npy")
    
    num_cores = multiprocessing.cpu_count() - 4 # save 4 core for some other processes
    inputs = tqdm(imlist)
    print(f"Processing {len(imlist)} in {num_cores} cores")
    processed_list = Parallel(n_jobs=num_cores)(delayed(alignment.get_coefs_im)(i, save_path, log_dir, n_coef=128, func=get_coef_fun) for i in inputs)
    with open(f'{log_dir}/images_fft_done.pkl', 'wb') as success_list:
        pickle.dump(processed_list, success_list)

def calculate_fft_hpa():
    cell_line = "U-2 OS"
    save_dir = f"/data/2Dshapespace/{cell_line.replace(' ','_')}/cell_masks"
    d = Path(save_dir)
    save_path = Path(f"/data/2Dshapespace/{cell_line.replace(' ','_')}/fftcoefs")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_dir = f"/data/2Dshapespace/{cell_line.replace(' ','_')}/logs"

    imlist= glob.glob(f"{save_dir}/*.npy")
    
    num_cores = multiprocessing.cpu_count() - 4 # save 4 core for some other processes
    inputs = tqdm(imlist)
    print(f"Processing {len(imlist)} in {num_cores} cores")
    processed_list = Parallel(n_jobs=num_cores)(delayed(alignment.get_coefs_im)(i, save_path, log_dir, n_coef=128, func=get_coef_fun) for i in inputs)
    with open(f'{log_dir}/images_fft_done.pkl', 'wb') as success_list:
        pickle.dump(processed_list, success_list)

def calculate_fft_ccd():
    dataset = "S-BIAD34"
    d = f"/data/2Dshapespace/{dataset}"
    sc_mask_dir = f"{d}/cell_masks"
    save_path = f"{d}/fftcoefs"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_dir = f"{d}/logs"

    abids = os.listdir(sc_mask_dir)
    imlist = [glob.glob(f"{sc_mask_dir}/{ab}/*.npy") for ab in abids]
    imlist = [item for sublist in imlist for item in sublist]
    
    num_cores = multiprocessing.cpu_count() - 4 # save 4 core for some other processes
    inputs = tqdm(imlist)
    print(f"Processing {len(imlist)} in {num_cores} cores")
    processed_list = Parallel(n_jobs=num_cores)(delayed(alignment.get_coefs_im)(i, save_path, log_dir, n_coef=128, func=get_coef_fun) for i in inputs)
    with open(f'{log_dir}/images_fft_done.pkl', 'wb') as success_list:
        pickle.dump(processed_list, success_list)

if __name__ == "__main__": 
    s_t = time.time()
    #calculate_fft_hpa()
    calculate_fft_ccd()
    print(f"Done in {np.round((time.time()-s_t)/3600,2)} h.")