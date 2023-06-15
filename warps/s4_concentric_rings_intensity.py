import os
import sys
sys.path.append("..")
from coefficients import coefs
from utils import plotting
from pathlib import Path
import numpy as np
from tqdm import tqdm
# import h5py
import argparse
import glob
import subprocess
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed

def grep(pattern, file_path):
    try:
        output = subprocess.check_output(['grep', pattern, file_path], universal_newlines=True)
        return output.splitlines()
    except subprocess.CalledProcessError:
        return []

def sample_intensity(l_num, cfg, args, shift_path, data_dir, protein_dir, mappings, inverse_func):
    data_ = l_num.strip().split(",")
    if len(data_[1:]) != cfg.N_COEFS * 4:
        return # continue
    sc_path = data_[0]
    img_id = os.path.basename(sc_path).replace(".npy","")
    if mappings[mappings.cell_idx==img_id].sc_target.values[0] in ["Negative","Multi-Location"]:
        #mappings.cell_idx.str.contains(img_id).sum() == 0: # Only single label cell
        #print(mappings.id.str.contains(img_id).sum())
        return #continue
    raw_protein_path = f"{data_dir}/{img_id}_protein.png"
    save_protein_path = f"{protein_dir}/{img_id}_protein.npy"
    #print(raw_protein_path, save_protein_path, img_id)
    if os.path.exists(save_protein_path):
        return# continue
    line_ = grep(img_id+".npy", shift_path)
    if line_ == []:
        print(f"{img_id} not found")
        return # continue
    data_shifts = line_[0].strip().split(";") 
    ori_fft = [
        complex(s.replace("i", "j")) for s in data_[1:]
    ]  # applymap(lambda s: complex(s.replace('i', 'j')))
    
    #print(sc_path, data_shifts)
    shifts = dict()
    shifts["theta"] = float(data_shifts[1])
    shifts["shift_c"] = (
            float(data_shifts[2].split(",")[0].strip("(")),
            (float(data_shifts[2].split(",")[1].strip(")"))),
        )
    shifts["sc_label"] = mappings[mappings.cell_idx==img_id].sc_target.values[0]
    
    if True:
        intensity = plotting.get_protein_intensity(
            pro_path=raw_protein_path,
            shift_dict=shifts,
            ori_fft=ori_fft,
            n_coef=cfg.N_COEFS,
            inverse_func=inverse_func,
            fourier_algo=cfg.COEF_FUNC,
            binarize=False,
            n_isos=args.n_isos
        )
        np.save(save_protein_path, intensity)
    if np.random.random() > 0.95:
        plotting.plot_interpolation3(
            shape_path=raw_protein_path.replace("_protein.png",".npy"),
            pro_path=raw_protein_path,
            shift_dict=shifts,
            save_path=save_protein_path.replace(".npy",".png"),
            ori_fft=ori_fft,
            reduced_fft=None,
            n_coef=cfg.N_COEFS,
            inverse_func=inverse_func,
            n_isos=args.n_isos
        )

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_line", type=str)
    parser.add_argument("--n_isos", nargs='+', type=int)
    args = parser.parse_args()
    import configs.config as cfg
    """
    num_cores = multiprocessing.cpu_count() -1 # save 1 core for some other processes
    pool = multiprocessing.Pool()
    processed_list = Parallel(n_jobs=num_cores)(delayed(myfunction)(i, im_df, mask_dir, image_dir, save_dir, log_dir) for i in inputs)

    n_cv = 10
    fourier_df = read_complex_df(fft_dir=fft_path, cfg.N_COEFS=128, n_cv=10, n_samples = 1000)
    """
    if cfg.COEF_FUNC == "fft":
        get_coef_fun = coefs.fourier_coeffs  # coefs.wavelet_coefs  #
        inverse_func = coefs.inverse_fft  # coefs.inverse_wavelet
    elif cfg.COEF_FUNC == "wavelet":
        get_coef_fun = coefs.wavelet_coefs
        inverse_func = coefs.inverse_wavelet

    cell_line = args.cell_line
    project_dir = os.path.join(os.path.dirname(cfg.PROJECT_DIR), cell_line)
    data_dir = f"{project_dir}/cell_masks"
    protein_dir = f"{project_dir}/sampled_intensity_bin"
    if not os.path.exists(protein_dir):
        os.makedirs(protein_dir)
    
    cellline_meta = os.path.join(project_dir, os.path.basename(cfg.META_PATH).replace(".csv", "_splitVesiclesPCP.csv"))
    mappings = pd.read_csv(cellline_meta)
    #mappings = mappings[~mappings.sc_target.isin(["Negative","Multi-Location"])]
    #print(mappings.sc_target.value_counts(), mappings.cell_idx)
    log_dir = f"{project_dir}/logs"
    fft_dir = f"{project_dir}/fftcoefs/{cfg.ALIGNMENT}"
    fft_path = os.path.join(fft_dir, f"fftcoefs_{cfg.N_COEFS}.txt")
    shift_path = os.path.join(fft_dir, f"shift_error_meta_fft{cfg.N_COEFS}.txt")
    with open(fft_path) as f:
        count = sum(1 for _ in f)
    with open(fft_path) as f_shift:
        count2 = sum(1 for _ in f_shift)
    print(count, count2)
    completed = [os.path.basename(img_id).replace("_protein.npy","") for img_id in glob.glob(f"{protein_dir}/*.npy")]
    print(len(completed),completed[:4])
    n_processes = multiprocessing.cpu_count() - 10
    chunk_size = count//n_processes
    print(f"processing {count} in {n_processes} cpu in chunk {chunk_size}")
    with open(fft_path, "r") as f: #, open(shift_path, "r") as f_shift:
        processed_list = Parallel(n_jobs=n_processes)(
                delayed(sample_intensity)(l_num, cfg, args, shift_path, data_dir, protein_dir, mappings, inverse_func,)
                for l_num in f)
        
if __name__ == "__main__":
    """
    main()
    """
    try:
        main()
        sys.exit(0)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("Exception:", exc_type.__name__)
        print("Traceback:")
        tb = exc_traceback
        while tb is not None:
            frame = tb.tb_frame
            line = tb.tb_lineno
            filename = frame.f_code.co_filename
            print(f"  File \"{filename}\", line {line}, in {frame.f_code.co_name}")
            tb = tb.tb_next        
        print("Message:", str(exc_value))
        sys.exit(1)
