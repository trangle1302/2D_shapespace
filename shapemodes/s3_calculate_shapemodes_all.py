import os
import sys
sys.path.append("..")
from shapemodes import dimreduction
from coefficients import coefs
from utils import plotting
from sklearn.decomposition import PCA, FastICA, IncrementalPCA
import pandas as pd
import numpy as np
import glob
import random
import json
import resource

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024, hard))


def get_memory():
    with open("/proc/meminfo", "r") as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                free_memory += int(sline[1])
    return free_memory

def load_fft(cfg, project_dir):
    fft_dir = f"{project_dir}/fftcoefs/{cfg.ALIGNMENT}"
    fft_path = os.path.join(fft_dir, f"fftcoefs_{cfg.N_COEFS}.txt")
    n_coef = cfg.N_COEFS
    n_samples = cfg.N_SAMPLES
    
    with open(fft_path) as f:
        count = sum(1 for _ in f)
    if (n_samples != -1) & (n_samples > count):
        n_samples = count

    for i in range(cfg.N_CV):
        with open(fft_path, "r") as file:
            lines = {}
            if n_samples == -1:
                specified_lines = range(count)
            else:
                specified_lines = random.sample(
                    range(count), n_samples
                )  # 10k cells/ CV
            print(f"Loading {len(specified_lines)} lines")
            # loop over lines in a file
            for pos, l_num in enumerate(file):
                # check if the line number is specified in the lines to read array
                if pos in specified_lines:
                    # print the required line number
                    data_ = l_num.strip().split(",")
                    if cfg.COEF_FUNC == "efd":
                        if len(data_[1:]) != (n_coef * 4 + 2) * 2:
                            continue
                    elif len(data_[1:]) != n_coef * 4:
                        continue
                    # data_dict = {data_dict[0]:data_dict[1:]}
                    lines[data_[0]] = data_[1:]
    return lines
                    
def main():
    import configs.config_all as cfg

    df_all = pd.DataFrame()
    mappings = pd.read_csv(cfg.META_PATH)
    labels = []
    for idx, cell_line in enumerate(cfg.CELL_LINE):
        project_dir = f"{cfg.PROJECT_DIR}/{cell_line.replace(' ','_')}"
        lines = load_fft(cfg, project_dir)

        cell_nu_ratio = pd.read_csv(f"{project_dir}/cell_nu_ratio.txt")
        cell_nu_ratio.columns = ["path", "name", "nu_area", "cell_area", "ratio"]
        if cell_line != "BJ":
            rm_cells = cell_nu_ratio[cell_nu_ratio.ratio > 8].name.to_list()
            lines = {
                k: lines[k]
                for k in lines.keys()
                if os.path.basename(k).split(".")[0] not in rm_cells
            }
        keep_cells = [cell_id.split("_", 1)[1] for cell_id in mappings.id]
        #print(f"Removing border cells leftover: {len(keep_cells)}")
        lines = {
            k: lines[k]
            for k in lines.keys()
            if os.path.basename(k).split(".")[0] in keep_cells
        }

        df = pd.DataFrame(lines).transpose()
        df_all = pd.concat([df_all, df], ignore_index=True)
        labels += np.repeat(idx, len(df)).tolist()
        print(f"After adding {cell_line}, current df shape: {df_all.shape}")
    df = df_all

    if cfg.COEF_FUNC == "fft":
        df = df.applymap(lambda s: complex(s.replace("i", "j")))

    if cfg.MODE == "nuclei":
        df = df.iloc[:, (df.shape[1] // 2) :]
    elif cfg.MODE == "cell":
        df = df.iloc[:, : (df.shape[1] // 2)]

    shape_mode_path = f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}"
    if not os.path.isdir(shape_mode_path):
        os.makedirs(shape_mode_path)

    use_complex = False
    if cfg.COEF_FUNC == "fft":
        get_coef_fun = coefs.fourier_coeffs
        inverse_func = coefs.inverse_fft
        if not use_complex:
            df_ = pd.concat(
                [
                    pd.DataFrame(np.matrix(df).real),
                    pd.DataFrame(np.matrix(df).imag),
                ],
                axis=1,
            )
            print(df.shape, df_.shape)
            pca = PCA()  # IncrementalPCA(whiten=True) #PCA()
            pca.fit(df_)
            plotting.display_scree_plot(pca, save_dir=shape_mode_path)
        else:
            df_ = df
            pca = dimreduction.ComplexPCA(n_components=df_.shape[1])
            pca.fit(df_)
            plotting.display_scree_plot(pca, save_dir=shape_mode_path)
    elif cfg.COEF_FUNC == "wavelet":
        get_coef_fun = coefs.wavelet_coefs
        inverse_func = coefs.inverse_wavelet
        df_ = df
        pca = PCA(n_components=df_.shape[1])
        pca.fit(df_)
        plotting.display_scree_plot(pca, save_dir=shape_mode_path)
    elif cfg.COEF_FUNC == "efd":
        get_coef_fun = coefs.elliptical_fourier_coeffs
        inverse_func = coefs.backward_efd
        df_ = df
        print(f"Number of samples {df_.shape[0]}, number of coefs {df_.shape[1]}")
        pca = PCA(n_components=df_.shape[1])
        pca.fit(df_)
        plotting.display_scree_plot(pca, save_dir=shape_mode_path)

    scree = pca.explained_variance_ratio_ * 100
    for percent in np.arange(70, 100, 5):
        n_pc = np.sum(scree.cumsum() < percent) + 1
        print(f"{n_pc} to explain {percent} % variance")
    n_pc = np.sum(scree.cumsum() < 95) + 1
    n_pc = 8 if n_pc < 8 else n_pc
    pc_names = [f"PC{c}" for c in range(1, 1 + len(pca.components_))]
    pc_keep = [f"PC{c}" for c in range(1, 1 + n_pc)]
    matrix_of_features_transform = pca.transform(df_)
    df_trans = pd.DataFrame(data=matrix_of_features_transform.copy())
    df_trans.columns = pc_names
    df_trans.index = df.index
    df_trans[list(set(pc_names) - set(pc_keep))] = 0
    print(matrix_of_features_transform.shape, df_trans.shape)
    
    # Cell density on major PC
    plotting.plot_pc_density(df_trans["PC1"], df_trans["PC2"], save_path=f"{shape_mode_path}/PC1vsPC2_cell_density.png")
    plotting.plot_pc_density(df_trans["PC2"], df_trans["PC3"], save_path=f"{shape_mode_path}/PC2vsPC3_cell_density.png")
    
    # Cell line distributions on PC1 vs PC2
    plotting.scatter_hist(df_trans["PC1"], df_trans["PC2"], labels, save_path=f"{shape_mode_path}/PC1vsPC2_scatter_hist.png")

    # Cell line distributions on PC2 vs PC3
    for pc in pc_keep:
        plotting.plot_histogram_with_fitted_line(df_trans[pc], labels, show_data=False, save_path=f"{shape_mode_path}/{pc}_scatter_hist.png")
        
    pm = plotting.PlotShapeModes(
        pca,
        df_trans,
        cfg.N_COEFS,
        pc_keep,
        scaler=None,
        complex_type=use_complex,
        fourier_algo=cfg.COEF_FUNC,
        inverse_func=inverse_func,
    )
    if cfg.MODE == "cell_nuclei":
        pm.plot_avg_cell(dark=False, save_dir=shape_mode_path)
    elif cfg.MODE == "nuclei":
        pm.plot_avg_nucleus(dark=False, save_dir=shape_mode_path)

    n_ = 10  # number of random cells to plot
    cells_assigned = dict()
    for pc in pc_keep:
        pm.plot_shape_variation_gif(pc, dark=False, save_dir=shape_mode_path)
        pm.plot_shape_variation(pc, dark=False, save_dir=shape_mode_path)
        pm.plot_pc_hist(pc, save_dir=shape_mode_path)
        pm.plot_pc_dist(pc, save_dir=shape_mode_path)

        pc_indexes_assigned, bin_links = pm.assign_cells(pc)

        # print(pc_indexes_assigned, len(pc_indexes_assigned))
        # print(bin_links, len(bin_links))
        # print([len(b) for b in bin_links])
        cells_assigned[pc] = [list(b) for b in bin_links]
            
    with open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json", "w") as fp:
        json.dump(cells_assigned, fp)


if __name__ == "__main__":
    memory_limit()  # Limitates maximun memory usage
    try:
        main()
    except MemoryError:
        sys.stderr.write("\n\nERROR: Memory Exception\n")
        sys.exit(1)
