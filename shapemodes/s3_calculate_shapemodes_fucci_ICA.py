import os
import sys
sys.path.append("..")
from shapemodes import dimreduction
from coefficients import coefs
from utils import plotting
from sklearn.decomposition import PCA,FastICA
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def calculate_shapemode(df, n_coef, mode, fun="fft", shape_mode_path="", fft_path=""):
    """ Calculate shapemodes based on the coefficient matrix given
    Parameters
        df: Matrix of coefficient
        n_coef: number of fourier coefficients
        mode: nuclei (shapemode of nuclei) or cell_nuclei (shapemode of both cell and nuclei)
        shape_mode_path: save dir
    Returns
        None
    """
    print(f"Saving to {shape_mode_path}")
    if not os.path.isdir(shape_mode_path):
        os.makedirs(shape_mode_path)
    use_complex = False
    get_coef_fun = coefs.fourier_coeffs
    inverse_func = coefs.inverse_fft
    df_ = pd.concat(
            [pd.DataFrame(np.matrix(df).real), pd.DataFrame(np.matrix(df).imag)],
            axis=1,
        )
    n_pc = 5#20
    ica = FastICA(n_components=n_pc, random_state=0)
    matrix_of_features_transform = ica.fit_transform(df_)
    pc_keep = [f"PC{c}" for c in range(1, 1 + n_pc)]
    df_trans = pd.DataFrame(data=matrix_of_features_transform.copy())
    df_trans.columns = pc_keep
    df_trans.index = df.index
    
    # Rank IC by Fraction of Explained Variance
    A_ = ica.mixing_   
    variance_explained = np.sum(A_**2, axis=0)
    variance_explained /= variance_explained.sum()
    #variance_order = np.argsort(variance_explained)
    #original_pair = sorted(zip(pc_keep, variance_explained))
    pc_order, variance_order = map(list, zip(*sorted(zip(pc_keep, variance_explained), key=lambda x: x[1], reverse=True)))
    print(variance_order)
    sorted_pairs = list(zip(pc_keep, variance_explained))
    # Extract the sorted items from the original list
    #sorted_pc = [item for item, _ in sorted_pairs]
    #sort_pc = sorted(pc_keep, key=lambda x: variance_order.index(pc_keep.index(x)))
    print('PC ordered by variance: ',pc_order, sorted_pairs)
    
    # Cell density on major PC
    plotting.plot_pc_density(df_trans["PC1"], df_trans["PC2"], save_path=f"{shape_mode_path}/PC1vsPC2_cell_density.png")
    plotting.plot_pc_density(df_trans["PC2"], df_trans["PC3"], save_path=f"{shape_mode_path}/PC2vsPC3_cell_density.png")

    # cheat to correct formatting
    setattr(ica, 'explained_variance_ratio_', [0,0,0])

    pm = plotting.PlotShapeModes(
        ica,
        df_trans,
        n_coef,
        pc_keep,
        scaler=None,
        complex_type=use_complex,
        fourier_algo=fun,
        inverse_func=inverse_func,
        mode=mode,
    )
    if mode == "cell_nuclei":
        pm.plot_avg_cell(dark=False, save_dir=shape_mode_path)
    elif mode == "nuclei":
        pm.plot_avg_nucleus(dark=False, save_dir=shape_mode_path)

    cells_assigned = dict()
    for pc in pc_keep:
        pm.plot_shape_variation_gif(pc, dark=False, save_dir=shape_mode_path)
        pm.plot_pc_dist(pc, save_dir=shape_mode_path)
        pm.plot_pc_hist(pc, save_dir=shape_mode_path)
        pm.plot_shape_variation(pc, dark=False, save_dir=shape_mode_path)
        pc_indexes_assigned, bin_links = pm.assign_cells(pc)
        cells_assigned[pc] = [list(b) for b in bin_links]
        #print(cells_assigned[pc][:3])
    
        plotting.plot_example_cells(bin_links, 
                                    n_coef=n_coef, 
                                    cells_per_bin=5, 
                                    shape_coef_path=fft_path, 
                                    save_path=f"{shape_mode_path}/{pc}_example_cells.png")


    with open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json", "w") as fp:
        json.dump(cells_assigned, fp)
    return df_trans

def main():
    import configs.config as cfg

    n_samples = cfg.N_SAMPLES
    fft_dir = f"{cfg.PROJECT_DIR}/fftcoefs/{cfg.ALIGNMENT}"
    log_dir = f"{cfg.PROJECT_DIR}/logs"
    fft_path = os.path.join(fft_dir, f"fftcoefs_{cfg.N_COEFS}.txt")

    all_locations = dict((v, k) for k, v in cfg.LABEL_TO_ALIAS.items())

    with open(fft_path) as f:
        count = sum(1 for _ in f)

    for i in range(cfg.N_CV):
        with open(fft_path, "r") as file:
            lines = dict()
            if n_samples == -1:
                specified_lines = range(count)
            else:
                specified_lines = random.sample(
                    range(count), n_samples
                )  # 10k cells/ CV
            # loop over lines in a file
            for pos, l_num in enumerate(file):
                # check if the line number is specified in the lines to read array
                # print(pos)
                if pos in specified_lines:
                    # print the required line number
                    data_ = l_num.strip().split(",")
                    if cfg.COEF_FUNC == "efd":
                        if len(data_[1:]) != (cfg.N_COEFS * 4 + 2) * 2:
                            continue
                    elif len(data_[1:]) != cfg.N_COEFS * 4:
                        continue
                    # data_dict = {data_dict[0]:data_dict[1:]}
                    lines[data_[0]] = data_[1:]

        df = pd.DataFrame(lines).transpose()
        print(df.shape)
        if cfg.COEF_FUNC == "fft":
            df = df.applymap(lambda s: complex(s.replace("i", "j")))

        if cfg.MODE == "nuclei":
            df = df.iloc[:, (df.shape[1] // 2) :]
        elif cfg.MODE == "cell":
            df = df.iloc[:, : (df.shape[1] // 2)]
        print(cfg.CELL_LINE, cfg.ALIGNMENT, cfg.MODE, df.shape)
        df["matchid"] = [
            k.replace("/data/2Dshapespace/S-BIAD34/cell_masks/", "").replace(
                ".npy", ""
            )
            for k in df.index
        ]

        df_ = df.drop(columns=["matchid"])
        shape_mode_path = f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}_ICA_5components"
        df_trans = calculate_shapemode(
            df_,
            cfg.N_COEFS,
            cfg.MODE,
            fun=cfg.COEF_FUNC,
            shape_mode_path=shape_mode_path,
            fft_path=fft_path
        )
        df_trans["matchid"] = df["matchid"] 
        # Shape modes of G1, G2/S, G2 cells:
        sc_stats = pd.read_csv(f"{cfg.PROJECT_DIR}/single_cell_statistics.csv")
        sc_stats["matchid"] = sc_stats.ab_id + "/" + sc_stats.cell_id
        #print(sc_stats.matchid[:3])
        
        df_trans = df_trans.merge(sc_stats, on="matchid")
        df_trans.to_csv(f"{shape_mode_path}/transformed_matrix.csv")
        plt.scatter(df_trans["PC1"], df_trans["PC2"], c=df_trans["pseudotime"], cmap='RdYlGn', alpha=0.1)    
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Pseudotime')
        # Add labels and title
        plt.xlabel('PC1')
        plt.ylabel('PC2')
if __name__ == "__main__":
    #memory_limit()  # Limitates maximun memory usage
    try:
        main()
        sys.exit(0)
    except MemoryError:
        sys.stderr.write("\n\nERROR: Memory Exception\n")
        sys.exit(1)
