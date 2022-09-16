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
from utils.complex import read_complex_df

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
    n_samples = 300000
    n_cv = 10
    cell_line = "U-2 OS" #"S-BIAD34"#"U-2 OS"
    project_dir = "/data/2Dshapespace"
    protein_dir = f"{project_dir}/{cell_line.replace(' ','_')}/sampled_intensity"
    if not os.path.exists(protein_dir):
        os.makedirs(protein_dir)

    log_dir = f"{project_dir}/{cell_line.replace(' ','_')}/logs"
    fft_dir = f"{project_dir}/{cell_line.replace(' ','_')}/fftcoefs"
    fft_path = os.path.join(fft_dir,f"fftcoefs_{n_coef}.txt")
    shift_path = os.path.join(fft_dir,f"shift_error_meta_fft{n_coef}.txt")
    with open(fft_path) as f:
        count = sum(1 for _ in f)
        
    with open(fft_path, "r") as f, open(shift_path, "r") as f_shift:
        for pos, (l_num, l_shift) in enumerate(tqdm(zip(f,f_shift), total=count)):
            data_ = l_num.strip().split(',')
            if len(data_[1:]) != n_coef*4:
                continue
            sc_path = data_[0]
            ori_fft = [np.complex(s.replace('i', 'j')) for s in data_[1:]]# applymap(lambda s: np.complex(s.replace('i', 'j'))) 
            
            #print(ori_fft, sc_path)

            data_shifts = l_shift.strip().split(',')
            assert sc_path == data_shifts[0]
            shifts = dict()
            shifts["theta"] = np.float(data_shifts[1])
            shifts["shift_c"] = (np.float(data_shifts[2].strip('(')),(np.float(data_shifts[3].strip(')'))))
            #print(shifts)
            protein_path = Path(str(sc_path).replace(".npy","_protein.png"))
            intensity = plotting.get_protein_intensity(
                                pro_path = protein_path, 
                                shift_dict = shifts,
                                ori_fft = ori_fft, 
                                n_coef = n_coef, 
                                inverse_func = inverse_func
                                )
            np.save(f"{protein_dir}/{Path(protein_path).stem}", intensity)
            # np.load('/data/2Dshapespace/U-2_OS/sampled_intensity/1118_F1_2_2_protein.npy')
            
"""
    if False:
        for l in ls:
            if l in list(df_sl_Label.Link):
                #print(l)
                protein_path = Path(str(l).replace(".npy","_protein.png"))
                ori_fft = df.loc[df.index== l].values[0]

                intensity = plotting.get_protein_intensity(
                    pro_path = protein_path, 
                    shift_dict = shifts[l],
                    ori_fft = ori_fft, 
                    n_coef = n_coef, 
                    inverse_func = inverse_func
                    )
        
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
                        
            for PC in pc_keep:
            #df_sl_Label = mappings[mappings.sc_locations_reindex == LABELINDEX]
                pc1, pc1l = pm.assign_cells(PC)
                
                #pc1l_Nucleoplasm = [l for l in ls for ls in pc1l if l in df_sl_Nucleoplasm.Link]
                
                shape = (21,256)
                intensities__pc1 = []
                counts = []
                for ls in pc1l:
                    intensities = []
                    i= 0
                    for l in ls:
                        if l in list(df_sl_Label.Link):
                            #print(l)
                            protein_path = Path(str(l).replace(".npy","_protein.png"))
                            ori_fft = df.loc[df.index== l].values[0]

                            intensity = plotting.get_protein_intensity(
                                pro_path = protein_path, 
                                shift_dict = shifts[l],
                                ori_fft = ori_fft, 
                                n_coef = n_coef, 
                                inverse_func = inverse_func
                                )
                            
                            #fig, ax = plt.subplots()
                            #plt.imshow(intensity)
                            intensities += [intensity.flatten()]
                            i +=1
                    counts += [i]
                    if len(intensities) == 0:
                        print('No cell sample at this bin for Nucleoplasm')
                        intensities__pc1 += [np.zeros(shape)]
                    else:
                        print(len(intensities))
                        intensities__pc1 += [np.nanmean(intensities, axis=0).reshape(intensity.shape)]
                
                pm.protein_intensities = intensities__pc1/np.array(intensities__pc1).max()
                pm.plot_protein_through_shape_variation_gif(PC)
"""