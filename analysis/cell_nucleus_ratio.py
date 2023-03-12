import os
import numpy as np
import pandas as pd
import glob
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import time

def check_nucleus_cell_size(image_path, save_dir):
    nu_cell_array = np.load(image_path)
    nu_area = np.sum(nu_cell_array[1,:,:] >0)
    cell_area = np.sum(nu_cell_array[0,:,:] >0)
    cell_nu_ratio = cell_area / nu_area
    image_name = os.path.basename(image_path).split(".")[0]
    with open(f"{save_dir}/cell_nu_ratio.txt","a") as f:
        f.write(",".join(map(str,[image_path, image_name, nu_area, cell_area, cell_nu_ratio])) + '\n')
    """
    if cell_nu_ratio > 8:
        with open(f"{save_dir}/bad_cell_list.txt", "a") as F:
            # Saving: image_name
            image_name = os.path.basename(image_path).split(".")[0]
            F.write(",".join(map(str,[image_path,image_name,])) + '\n')
        return True
    else:
        return False
    """

def main():
    s = time.time()
    mask_dir = "/data/2Dshapespace/S-BIAD34/cell_masks2" #"/data/2Dshapespace/U-2_OS/cell_masks"
    save_dir = "/data/2Dshapespace/S-BIAD34" # "/data/2Dshapespace/U-2_OS/"
    imlist = glob.glob(f"{mask_dir}/*.npy")
    if len(imlist)==0:
        imlist = glob.glob(f"{mask_dir}/*/*.npy")
    print(f"{len(imlist)} cells found")
    num_cores = multiprocessing.cpu_count() - 15 # save 4 core for some other processes
    inputs = tqdm(imlist)
    print(f"Processing {len(imlist)} in {num_cores} cores")
    processed_list = Parallel(n_jobs=num_cores)(delayed(check_nucleus_cell_size)(i, save_dir) for i in inputs)
    print(f"Finished in {(time.time() - s)/60} min")

if __name__ == "__main__":
    main()
