import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from skimage.metrics import structural_similarity
from warps import parameterize

LABEL_TO_ALIAS = {
    0: "Nucleoplasm",
    1: "NuclearM",
    2: "Nucleoli",
    3: "NucleoliFC",
    4: "NuclearS",
    5: "NuclearB",
    6: "EndoplasmicR",
    7: "GolgiA",
    8: "IntermediateF",
    9: "ActinF",
    10: "Microtubules",
    # 11: 'MitoticS',
    12: "Centrosome",
    13: "PlasmaM",
    14: "Mitochondria",
    # 15: 'Aggresome',
    16: "Cytosol",
    17: "VesiclesPCP",
    # 18: 'Negative',
    # 19:'Multi-Location',
}


def correlation(value_dict, method_func):
    cor_mat = np.zeros((len(value_dict), len(value_dict)))
    for i, (k1, v1) in enumerate(value_dict.items()):
        for j, (k2, v2) in enumerate(value_dict.items()):
            cor_mat[i, j] = method_func(v1, v2)
    return cor_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", help="Organelle class", type=str)
    args = parser.parse_args()
    print(args.org)

    n_coef = 128
    cell_line = "U-2 OS"
    project_dir = f"/scratch/users/tle1302/2Dshapespace/{cell_line.replace(' ','_')}"
    log_dir = f"{project_dir}/logs"
    fftcoefs_dir = f"{project_dir}/fftcoefs/fft_major_axis_polarized"
    fft_path = os.path.join(fftcoefs_dir, f"fftcoefs_{n_coef}.txt")
    shape_mode_path = (
        f"{project_dir}/shapemode/{cell_line.replace(' ','_')}/fft_major_axis_polarized"
    )
    avg_organelle_dir = f"{project_dir}/morphed_protein_avg"

    merged_bins = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    # Panel 1: Organelle through shapespace
    for PC in np.arange(8):
        for org in LABEL_TO_ALIAS.values():
            for i, bin_ in enumerate(merged_bins):
                if len(bin_) == 1:
                    bin_ = bin_[0]
                    org_bin = imread(f"{avg_organelle_dir}/PC{PC}/bin_{bin_}_{org}")

    # Panel 2: Organelle heatmap through shapespace
    for PC in np.arange(8):
        for b in np.arange(11):
            images = {}
            for i, bin_ in enumerate(merged_bins):
                if len(bin_) == 1:
                    b = bin_[0]
                    images = {}
                    for org in LABEL_TO_ALIAS.values():
                        images[org] = imread(f"{d}/{PC}/bin{b}_{org}.png")

            ssim_scores = correlation(images, structural_similarity)
            ssim_df = pd.DataFrame(ssim_scores, columns=list(images.keys()))
            ssim_df.index = list(images.keys())
