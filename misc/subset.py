import os
import glob
import shutil
import numpy as np
import pandas as pd


def find_line(txt_path, specific_text):
    """
    Find line containing specific text in a large txt file

    Parameters
    ----------
    txt_path : str
        path to txt file.
    specific_text : str
        text to find.

    Returns
    -------
    l : str
        first line containing specific_text.
    index : int
        index of this line in the file
    """
    with open(txt_path, "r") as fp:
        lines = fp.readlines()
        for l in lines:
            if l.find(specific_text) != -1:
                return l, lines.index(l)


def subset_file(fft_path, imlist, check_len=True, n_coef=128):
    result_lines = dict()
    file = open(fft_path, "r")
    lines = file.readlines()
    for l in lines:
        data_ = l.strip().split(",")
        if check_len:
            if len(data_[1:]) != n_coef * 4:
                continue
        if data_[0] in imlist:
            result_lines[data_[0]] = data_[1:]
    return pd.DataFrame(result_lines)


def get_subset_masks_coefs(project_dir, n_sample=1000):
    np.random.seed(123)
    mask_dir = f"{project_dir}/cell_masks"
    coefs_dir = f"{project_dir}/fftcoefs"

    save_dir = f"{project_dir}/subset"
    if not os.path.exists(f"{save_dir}/cell_masks"):
        os.makedirs(f"{save_dir}/cell_masks")
    if not os.path.exists(f"{save_dir}/fftcoefs"):
        os.makedirs(f"{save_dir}/fftcoefs")

    imlist = glob.glob(f"{mask_dir}/*.npy")
    print(f"Found {len(imlist)} masks")
    print(imlist[:3])
    sub_imlist = glob.glob(f"{save_dir}/cell_masks/*.npy")
    sub_imlist = [
        f"{project_dir}/cell_masks/{os.path.basename(im)}" for im in sub_imlist
    ]
    if len(sub_imlist) < 900:
        sub_imlist = np.random.choice(imlist, n_sample)
        print(sub_imlist[:3])

        for img in sub_imlist:
            print(img, f"{save_dir}/cell_masks/{os.path.basename(img)}")
            shutil.copy(img, f"{save_dir}/cell_masks/{os.path.basename(img)}")

    df_fft = subset_file(f"{coefs_dir}/fftcoefs_128.txt", sub_imlist, check_len=True)
    df_fft = df_fft.transpose()
    df_fft.to_csv(f"{save_dir}/fftcoefs/fftcoefs_128.csv")
    print(f"{coefs_dir}/shift_error_meta_fft128.txt")
    df_meta = subset_file(
        f"{coefs_dir}/shift_error_meta_fft128.txt", sub_imlist, check_len=False
    )
    df_meta = df_meta.transpose()
    df_meta.to_csv(f"{save_dir}/fftcoefs/shift_error_meta_fft128.csv")


def main():
    project_dir = "/data/2Dshapespace/U-2_OS"
    get_subset_masks_coefs(project_dir, n_sample=1000)


if __name__ == "__main__":
    main()
