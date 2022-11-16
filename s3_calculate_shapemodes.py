import os
from utils.parameterize import get_coordinates
from utils import plotting, helpers, dimreduction, coefs, alignment
from sklearn.decomposition import PCA, IncrementalPCA
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
import json
import resource
import sys

LABEL_TO_ALIAS = {
  0: 'Nucleoplasm',
  1: 'NuclearM',
  2: 'Nucleoli',
  3: 'NucleoliFC',
  4: 'NuclearS',
  5: 'NuclearB',
  6: 'EndoplasmicR',
  7: 'GolgiA',
  8: 'IntermediateF',
  9: 'ActinF',
  10: 'Microtubules',
  11: 'MitoticS',
  12: 'Centrosome',
  13: 'PlasmaM',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'VesiclesPCP',
  19: 'Negative',
  19:'Multi-Location',
}

all_locations = dict((v, k) for k,v in LABEL_TO_ALIAS.items())
#%% Coefficients
fun = "efd"
if fun == "fft":
    get_coef_fun = coefs.fourier_coeffs 
    inverse_func = coefs.inverse_fft 
elif fun == "wavelet":
    get_coef_fun = coefs.wavelet_coefs
    inverse_func = coefs.inverse_wavelet
elif fun == "efd":
    get_coef_fun = coefs.elliptical_fourier_coeffs
    inverse_func = coefs.backward_efd

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    #resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def main():
    n_coef = 128
    n_samples = 10000
    n_cv = 2#0
    cell_line = "U-2 OS" #"S-BIAD34"#"U-2 OS"
    project_dir = f"/data/2Dshapespace/{cell_line.replace(' ','_')}"
    #project_dir = f"/scratch/users/tle1302/2Dshapespace/{cell_line.replace(' ','_')}" #"/data/2Dshapespace"
    #log_dir = f"{project_dir}/{cell_line.replace(' ','_')}/logs"
    #fft_dir = f"{project_dir}/{cell_line.replace(' ','_')}/fftcoefs"
    log_dir = f"{project_dir}/logs"
    fft_dir = f"{project_dir}/fftcoefs/{fun}"
    fft_path = os.path.join(fft_dir, f"fftcoefs_{n_coef}.txt")
    
    sampled_intensity_dir = Path(f"{project_dir}/sampled_intensity") #Path(f"/data/2Dshapespace/{cell_line.replace(' ','_')}/sampled_intensity")
    #mappings = pd.read_csv("/scratch/users/tle1302/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border.csv")
    mappings = pd.read_csv(f"/data/kaggle-dataset/publicHPA_umap/results/webapp/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border.csv")
    mappings = mappings[mappings['atlas_name']=='U-2 OS']
    #print(mappings.target.value_counts())
    print(mappings.columns)
    id_with_intensity = glob.glob(f"{sampled_intensity_dir}/*.npy")
    mappings["Link"] =[f"{sampled_intensity_dir}/{id.split('_',1)[1]}_protein.npy" for id in mappings.id]
    mappings = mappings[mappings.Link.isin(id_with_intensity)]
    print(mappings.target.value_counts())

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
                    if fun == "efd":
                        if len(data_[1:]) != (n_coef*4+2)*2:
                            continue
                    elif len(data_[1:]) != n_coef*4:
                            continue
                    #data_dict = {data_dict[0]:data_dict[1:]}
                    lines[data_[0]]=data_[1:]

        cell_nu_ratio = pd.read_csv(f"{project_dir}/cell_nu_ratio.txt")
        cell_nu_ratio.columns = ["path", "name", "nu_area","cell_area", "ratio"]
        rm_cells = cell_nu_ratio[cell_nu_ratio.ratio > 8].name.to_list()
        print(f"Large cell-nu ratio cells to remove: {len(rm_cells)}") # 6264 cells for ratio 10, and 16410 for ratio 8
        lines = {k:lines[k] for k in lines.keys() if os.path.basename(k).split(".")[0] not in rm_cells}
        print(len(lines))
        keep_cells = [cell_id.split("_",1)[1] for cell_id in mappings.id]
        print(f"Removing border cells leftover") 
        lines = {k:lines[k] for k in lines.keys() if os.path.basename(k).split(".")[0] in keep_cells}
        print(lines.keys())
        df = pd.DataFrame(lines).transpose()
        print(df.shape)
        print(df.iloc[0][0])
        if fun == "fft":
            df = df.applymap(lambda s: np.complex(s.replace('i', 'j'))) 
        shape_mode_path = f"{project_dir}/shapemode/{cell_line.replace(' ','_')}/{fun}"
        if not os.path.isdir(shape_mode_path):
            os.makedirs(shape_mode_path)
        
        use_complex = False
        if fun == "fft":
            if not use_complex:
                df_ = pd.concat(
                    [pd.DataFrame(np.matrix(df).real), pd.DataFrame(np.matrix(df).imag)], axis=1
                )
                pca = PCA()#IncrementalPCA(whiten=True) #PCA()
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
        elif fun == "efd":
            df_ = df
            print(df_.head())
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
        cells_assigned = dict()
        for pc in pc_keep:
            pm.plot_shape_variation_gif(pc, dark=False, save_dir=shape_mode_path)
            pm.plot_pc_dist(pc)
            pm.plot_pc_hist(pc)
            pm.plot_shape_variation(pc, dark=False, save_dir=shape_mode_path)

            pc_indexes_assigned, bin_links = pm.assign_cells(pc) 
            #print(pc_indexes_assigned, len(pc_indexes_assigned))
            #print(bin_links, len(bin_links))
            #print([len(b) for b in bin_links])
            cells_assigned[pc] = [list(b) for b in bin_links]
        
        with open(f'{shape_mode_path}/cells_assigned_to_pc_bins.json', 'w') as fp:
            json.dump(cells_assigned, fp)
        
        """
        if not os.path.isdir(f"{project_dir}/shapemode/organelle"):
            os.makedirs(f"{project_dir}/shapemode/organelle")
        meta = []
        for org in list(all_locations.keys())[:-1]:
            df_sl_Label = mappings[mappings.target == org]
            
            for PC, pc_cells in cells_assigned.items():
                print(org, PC, len(pc_cells), len(pc_cells[0]))
                shape = (21, n_coef*2)
                intensities_pcX = []
                counts = []
                for ls in pc_cells:
                    intensities = []
                    i= 0
                    for l in ls:
                        l = str(sampled_intensity_dir) + "/"+ Path(l).stem + "_protein.npy"
                        if l in list(df_sl_Label.Link):
                            intensity = np.load(l)
                            dummy_threshold = intensity.max() // 3
                            intensity = np.where(intensity > dummy_threshold, 1, 0)
                            intensities += [intensity.flatten()]
                            i +=1
                    counts += [i]
                    if len(intensities) == 0:
                        print(f'No cell sample at this bin for {org}')
                        intensities_pcX += [np.zeros(shape)]
                    else:
                        print(len(intensities))
                        intensities_pcX += [np.nanmean(intensities, axis=0).reshape(intensity.shape)]
                print(counts)
                meta += [[org]+ counts]
                intensities_pcX = np.array(intensities_pcX)
                print(intensities_pcX.shape)
                np.save(f"{project_dir}/shapemode/organelle/{org}_{PC}_intensity", intensities_pcX)
                pm.protein_intensities = intensities_pcX/intensities_pcX.max()
                pm.plot_protein_through_shape_variation_gif(PC, title=org, dark=True, save_dir=f"{project_dir}/shapemode/organelle")

        meta = pd.DataFrame(meta)
        meta.columns = ["org"] +["".join(("n_bin",str(i))) for i in range(11)]
        print(meta)
        meta.to_csv(f"{project_dir}/shapemode/organelle/cells_per_bin.csv", index=False)
        """
        
if __name__ == "__main__": 
    memory_limit() # Limitates maximun memory usage
    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)
