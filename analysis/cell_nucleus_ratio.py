import os
import sys
sys.path.append("..")
import numpy as np
import imageio
import glob
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import time

def check_nucleus_cell_size(image_path, save_dir):
    try:
        nu_cell_array = np.load(image_path)
        nu_area = np.sum(nu_cell_array[1, :, :] > 0)
        cell_area = np.sum(nu_cell_array[0, :, :] > 0)
        cell_nu_ratio = cell_area / nu_area
        protein = imageio.imread(image_path.replace(".npy","_protein.png"))
        pr_sum = np.sum(protein*nu_cell_array[0, :, :])
        pr_nu_sum = np.sum(protein*nu_cell_array[1, :, :])
        image_name = os.path.basename(image_path).split(".")[0]
        line = f"{image_path},{image_name},{nu_area},{cell_area},{cell_nu_ratio},{pr_sum},{pr_nu_sum}\n"

        #with open(f"{save_dir}/cell_nu_ratio.txt", "a") as f:
        #    f.write(
        #        ",".join(
        #            map(str, [image_path, image_name, nu_area, cell_area, cell_nu_ratio, pr_sum, pr_nu_sum])
        #        )
        #        + "\n"
        #    )
        return line
    except:
        print(f"Error in {image_path}")
        with open(f"{save_dir}/np_pickle_failing_check.txt", "a") as f:
            f.write(f"{image_path}\n")

def main():
    s = time.time()
    import configs.config as cfg
    mask_dir = f"{cfg.PROJECT_DIR}/cell_masks"
    save_dir = cfg.PROJECT_DIR
    imlist = glob.glob(f"{mask_dir}/*.npy")
    imlist = [f for f in imlist if 'ref' not in f]
    if len(imlist) == 0:
        imlist = glob.glob(f"{mask_dir}/*/*.npy")
    imlist = [i for i in imlist if "_ref" not in i]
    print(f"{len(imlist)} cells found")
    num_cores = multiprocessing.cpu_count() - 15  # save 4 core for some other processes
    # inputs = tqdm(imlist)
    print(f"Processing {len(imlist)} in {num_cores} cores")
    with open(f"{save_dir}/cell_nu_ratio.txt", "a+") as f:
        f.write("image_path,image_name,nu_area,cell_area,cell_nu_ratio,Protein_cell_sum,Protein_nu_sum\n")
        for image_path in tqdm(imlist, total=len(imlist)):
            # line = check_nucleus_cell_size(img_path, save_dir)
            try:
                nu_cell_array = np.load(image_path)
                nu_area = np.sum(nu_cell_array[1, :, :] > 0)
                cell_area = np.sum(nu_cell_array[0, :, :] > 0)
                cell_nu_ratio = cell_area / nu_area
                protein = imageio.imread(image_path.replace(".npy","_protein.png"))
                pr_sum = np.sum(protein*nu_cell_array[0, :, :])
                pr_nu_sum = np.sum(protein*nu_cell_array[1, :, :])
                image_name = os.path.basename(image_path).split(".")[0]
                line = f"{image_path},{image_name},{nu_area},{cell_area},{cell_nu_ratio},{pr_sum},{pr_nu_sum}\n"

                #with open(f"{save_dir}/cell_nu_ratio.txt", "a") as f:
                #    f.write(
                #        ",".join(
                #            map(str, [image_path, image_name, nu_area, cell_area, cell_nu_ratio, pr_sum, pr_nu_sum])
                #        )
                #        + "\n"
                #    )
                f.write(line)
            except:
                print(f"Error in {image_path}")
                with open(f"{save_dir}/np_pickle_failing_check.txt", "a") as ff:
                    ff.write(f"{image_path}\n")
            
    #processed_list = Parallel(n_jobs=num_cores)(
    #    delayed(check_nucleus_cell_size)(i, save_dir) for i in inputs
    #)
    print(f"Finished in {(time.time() - s)/60} min")

if __name__ == "__main__":
    main()
