import os
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import glob
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import time

def check_nucleus_cell_size(image_path, save_dir):
    nu_cell_array = np.load(image_path)
    nu_area = np.sum(nu_cell_array[1, :, :] > 0)
    cell_area = np.sum(nu_cell_array[0, :, :] > 0)
    cell_nu_ratio = cell_area / nu_area
    image_name = os.path.basename(image_path).split(".")[0]
    with open(f"{save_dir}/cell_nu_ratio.txt", "a") as f:
        f.write(
            ",".join(
                map(str, [image_path, image_name, nu_area, cell_area, cell_nu_ratio])
            )
            + "\n"
        )

def main():
    s = time.time()
    import configs.config as cfg
    mask_dir = f"{cfg.PROJECT_DIR}/cell_masks"
    save_dir = cfg.PROJECT_DIR
    imlist = glob.glob(f"{mask_dir}/*.npy")
    if len(imlist) == 0:
        imlist = glob.glob(f"{mask_dir}/*/*.npy")
    print(f"{len(imlist)} cells found")
    num_cores = multiprocessing.cpu_count() - 15  # save 4 core for some other processes
    inputs = tqdm(imlist)
    print(f"Processing {len(imlist)} in {num_cores} cores")
    processed_list = Parallel(n_jobs=num_cores)(
        delayed(check_nucleus_cell_size)(i, save_dir) for i in inputs
    )
    print(f"Finished in {(time.time() - s)/60} min")

if __name__ == "__main__":
    main()
