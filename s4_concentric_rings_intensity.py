import os
from coefs import coefs
from utils import plotting
from pathlib import Path
import numpy as np
from tqdm import tqdm
#import h5py

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
    
    """
    num_cores = multiprocessing.cpu_count() -1 # save 1 core for some other processes
    pool = multiprocessing.Pool()
    processed_list = Parallel(n_jobs=num_cores)(delayed(myfunction)(i, im_df, mask_dir, image_dir, save_dir, log_dir) for i in inputs)

    fourier_df = read_complex_df(fft_dir=fft_path, n_coef=128, n_cv=10, n_samples = 1000)
    """
    n_coef = 128
    n_cv = 10
    cell_line = "U-2 OS" #"S-BIAD34"#"U-2 OS"
    project_dir = "/data/2Dshapespace"
    protein_dir = f"{project_dir}/{cell_line.replace(' ','_')}/sampled_intensity2"
    if not os.path.exists(protein_dir):
        os.makedirs(protein_dir)

    log_dir = f"{project_dir}/{cell_line.replace(' ','_')}/logs"
    fft_dir = f"{project_dir}/{cell_line.replace(' ','_')}/fftcoefs"
    fft_path = os.path.join(fft_dir,f"fftcoefs_{n_coef}.txt")
    shift_path = os.path.join(fft_dir,f"shift_error_meta_fft{n_coef}.txt")
    with open(fft_path) as f:
        count = sum(1 for _ in f)
    with open(fft_path) as f_shift:
        count2 = sum(1 for _ in f_shift)
    print(count,count2)
    with open(fft_path, "r") as f, open(shift_path, "r") as f_shift:
        #for pos, (l_num, l_shift) in enumerate(tqdm(zip(f,f_shift), total=count)):
            #for l_shift in f_shift:
        for l_num in tqdm(f, total=count):
            data_ = l_num.strip().split(',')
            if len(data_[1:]) != n_coef*4:
                continue
            sc_path = data_[0]
            protein_path = Path(str(sc_path).replace(".npy","_protein.png"))            
            if os.path.exists(f"{protein_dir}/{Path(protein_path).stem}.npy"):
                continue
            
            for l_shift in f_shift:
                if l_shift.startswith(sc_path):
                    data_shifts = l_shift.strip().split(',')
                    break
                else:
                    pass 

            if sc_path != data_shifts[0]:
                continue
            assert sc_path == data_shifts[0]

            ori_fft = [np.complex(s.replace('i', 'j')) for s in data_[1:]]# applymap(lambda s: np.complex(s.replace('i', 'j'))) 
            
            shifts = dict()
            shifts["theta"] = np.float(data_shifts[1])
            shifts["shift_c"] = (np.float(data_shifts[2].strip('(')),(np.float(data_shifts[3].strip(')'))))
            #print(shifts)
            intensity = plotting.get_protein_intensity(
                                pro_path = protein_path, 
                                shift_dict = shifts,
                                ori_fft = ori_fft, 
                                n_coef = n_coef, 
                                inverse_func = inverse_func
                                )
            np.save(f"{protein_dir}/{Path(protein_path).stem}", intensity)
            # np.load('/data/2Dshapespace/U-2_OS/sampled_intensity/1118_F1_2_2_protein.npy')
    #h5f = h5py.File(f"{protein_dir}/{cell_line.replace(' ','_')}.h5","w")
    #ds_ = f.create_dataset("data", (img_src.count, img_src.shape[0], img_src.shape[1]), dtype='uint8', chunks=(1,512,512), compression='gzip', compression_opts=9)
