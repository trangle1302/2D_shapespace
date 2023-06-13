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
    with open(fft_path, "r") as f: #, open(shift_path, "r") as f_shift:
        # for pos, (l_num, l_shift) in enumerate(tqdm(zip(f,f_shift), total=count)):
        # for l_shift in f_shift:
        for l_num in tqdm(f, total=count):
            data_ = l_num.strip().split(",")
            if len(data_[1:]) != cfg.N_COEFS * 4:
                continue
            sc_path = data_[0]
            img_id = os.path.basename(sc_path).replace(".npy","")
            raw_protein_path = f"{data_dir}/{img_id}_protein.png"
            save_protein_path = f"{protein_dir}/{img_id}_protein.npy"
            #print(raw_protein_path, save_protein_path, img_id)
            if os.path.exists(save_protein_path):
                continue
         
            data_shifts = None
            with open(shift_path, "r") as f_shift: 
                for line in f_shift:
                    #print(line.find(img_id))
                    if img_id+".npy" in line: #line.find(img_id) != -1:
                        #print(line, img_id)
                        data_shifts = line.strip().split(";")
                        break
            if data_shifts == None:
                print(f"{img_id} not found")
                continue
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
            # np.load('/data/2Dshapespace/U-2_OS/sampled_intensity/1118_F1_2_2_protein.npy')
    # h5f = h5py.File(f"{protein_dir}/{cfg.CELL_LINE.replace(' ','_')}.h5","w")
    # ds_ = f.create_dataset("data", (img_src.count, img_src.shape[0], img_src.shape[1]), dtype='uint8', chunks=(1,512,512), compression='gzip', compression_opts=9)

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
