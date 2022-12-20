import os
import sys
sys.path.append("..") 
from warps.parameterize import get_coordinates
from coefficients import alignment, coefs
from pathlib import Path
import numpy as np
import glob
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import time


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

def calculate_fft_hpa():
    cell_line = "U-2 OS"
    save_dir = f"/data/2Dshapespace/{cell_line.replace(' ','_')}/cell_masks"
    d = Path(save_dir)
    save_path = Path(f"/data/2Dshapespace/{cell_line.replace(' ','_')}/fftcoefs/fft_major_axis_polarized_ud")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_dir = f"/data/2Dshapespace/{cell_line.replace(' ','_')}/logs"

    imlist= glob.glob(f"{save_dir}/*.npy")
    imlist = [im for im in imlist if os.path.getsize(im)>0]
    
    num_cores = multiprocessing.cpu_count() - 4 # save 4 core for some other processes
    inputs = tqdm(imlist)
    print(f"Processing {len(imlist)} in {num_cores} cores")
    processed_list = Parallel(n_jobs=num_cores)(delayed(alignment.get_coefs_im)(i, save_path, log_dir, n_coef=128, func=get_coef_fun, plot=np.random.choice([True,False], p=[0.001,0.999])) for i in inputs)
    with open(f'{log_dir}/images_fft_done.pkl', 'wb') as success_list:
        pickle.dump(processed_list, success_list)

def calculate_fft_ccd():
    dataset = "S-BIAD34"
    d = f"/data/2Dshapespace/{dataset}"
    sc_mask_dir = f"{d}/cell_masks"
    save_path = f"{d}/fftcoefs/fft_major_axis_polarized_ud"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_dir = f"{d}/logs"
    if True:
        abids = os.listdir(sc_mask_dir)
        imlist = [glob.glob(f"{sc_mask_dir}/{ab}/*.npy") for ab in abids]
        imlist = [item for sublist in imlist for item in sublist]
        imlist = [im for im in imlist if os.path.getsize(im)>0]
    if False:
        import pandas as pd
        imlist = pd.read_csv(f"{d}/failed_img.csv").iloc[:,0].values.tolist()
    num_cores = multiprocessing.cpu_count() - 10 # save 10 core for some other processes
    inputs = tqdm(imlist)
    print(f"Processing {len(imlist)} in {num_cores} cores, saving to {save_path}")
    processed_list = Parallel(n_jobs=num_cores)(delayed(alignment.get_coefs_im)(i, save_path, log_dir, n_coef=128, func=get_coef_fun, plot=np.random.choice([True,False], p=[0.001,0.999])) for i in inputs)
    with open(f'{log_dir}/images_fft_done.pkl', 'wb') as success_list:
        pickle.dump(processed_list, success_list)

if __name__ == "__main__": 
    s_t = time.time()
    #calculate_fft_hpa()
    calculate_fft_ccd()
    print(f"Done in {np.round((time.time()-s_t)/3600,2)} h.")
