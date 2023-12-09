import os
import sys
sys.path.append("..")
from coefficients import coefs
from utils import plotting
import matplotlib.pyplot as plt
import numpy as np
from utils import helpers
from utils.helpers import grep, get_line
import argparse
import glob
import subprocess
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from warps import image_warp_new as image_warp
from scipy.ndimage import center_of_mass, rotate
from skimage.transform import resize
from imageio import imread,imwrite
from tqdm import tqdm

def avg_cell_landmarks(ix_n, iy_n, ix_c, iy_c, n_landmarks=32):
    nu_centroid = helpers.find_centroid([(x_, y_) for x_, y_ in zip(ix_n, iy_n)])
    nu_centroid = [nu_centroid[0], nu_centroid[1]]
    print(f"Nucleus centroid of the avg shape: {nu_centroid}")

    # Move average shape from zero-centered coords to min=[0,0]
    min_x = np.min(ix_c)
    min_y = np.min(iy_c)
    nu_centroid[0] -= min_x
    nu_centroid[1] -= min_y
    ix_n -= min_x
    iy_n -= min_y
    ix_c -= min_x
    iy_c -= min_y

    if len(ix_n) != n_landmarks:
        ix_n, iy_n = helpers.equidistance(ix_n, iy_n, n_points=n_landmarks + 1)
        ix_c, iy_c = helpers.equidistance(ix_c, iy_c, n_points=n_landmarks * 2 + 1)
    nu_contour = np.stack([ix_n, iy_n]).T[:-1]
    cell_contour = np.stack([ix_c, iy_c]).T[:-1]
    # print(nu_contour.shape, cell_contour.shape)

    pts_avg = np.vstack(
        [
            np.asarray(nu_centroid),
            helpers.realign_contour_startpoint(nu_contour),
            helpers.realign_contour_startpoint(cell_contour, nearest_p=nu_centroid),
        ]
    )
    # print(pts_avg.max(), pts_avg.min(), cell_contour[:,0].max(), cell_contour[:,1].max())
    shape_x, shape_y = (
        np.round(cell_contour[:, 0].max()).astype("int"),
        np.round(cell_contour[:, 1].max()).astype("int"),
    )

    return pts_avg, (shape_x, shape_y)

def image_warping(l_num, cfg, shift_path, data_dir, protein_dir, mappings, pts_avg, shape_x, shape_y):
    data_ = l_num.strip().split(",")
    if len(data_[1:]) != cfg.N_COEFS * 4:
        if data_[0].contains('589_B4_3_'):
            print(data_[0])
        return # continue
    sc_path = data_[0]
    img_id = os.path.basename(sc_path).replace(".npy","")
    #print(mappings.cell_idx.tolist())
    if img_id == '589_B4_3_7':
        print(mappings[mappings.cell_idx==img_id].sc_target)
    if not img_id in mappings.cell_idx.tolist():
        return
    raw_protein_path = f"{data_dir}/{img_id}_protein.png"
    save_protein_path = f"{protein_dir}/{img_id}_protein.png"
    #print(raw_protein_path, save_protein_path, img_id)
    if os.path.exists(save_protein_path):
        return 
    #print('Saving to file: ', save_protein_path) 
    #print(mappings[mappings.cell_idx==img_id].sc_target.values[0])
    #if mappings[mappings.cell_idx==img_id].locations.values[0] not in cfg.ORGANELLES_FULLNAME:
    if mappings[mappings.cell_idx==img_id].sc_target.values[0] in ["Negative","Multi-Location"]:
        #mappings.cell_idx.str.contains(img_id).sum() == 0: # Only single label cell
        return
    #print(img_id)
    line_ = grep(img_id+".npy", shift_path)
    if line_ == []:
        print(f"{img_id} not found")
        return 
    #print(f"processing {img_id}")
    data_shifts = line_[0].strip().split(";") 
    ori_fft = [
        complex(s.replace("i", "j")) for s in data_[1:]
    ]  # applymap(lambda s: complex(s.replace('i', 'j')))
    
    
    shifts = dict()
    shifts["theta"] = float(data_shifts[1])
    shifts["shift_c"] = (
            float(data_shifts[2].split(",")[0].strip("(")),
            (float(data_shifts[2].split(",")[1].strip(")"))),
        )
    #print(mappings[mappings.cell_idx==img_id].sc_target.values)
    shifts["sc_label"] = mappings[mappings.cell_idx==img_id].sc_target.values[0]
    
    
    cell_shape = np.load(f"{data_dir}/{img_id}.npy")
    img = imread(f"{data_dir}/{img_id}_protein.png")
    if img.dtype == "uint16":
        img = (img / 256).astype(np.uint8)
    # print("Original image: ", img.max(), img.dtype, len(ls_))
    img = rotate(img, shifts["theta"])
    nu_ = rotate(cell_shape[1, :, :], shifts["theta"])
    cell_ = rotate(cell_shape[0, :, :], shifts["theta"])

    center_cell = center_of_mass(cell_)
    center_nuclei = center_of_mass(nu_)
    if (
        center_cell[1] > center_nuclei[1]
    ):  # Move 1 quadrant counter-clockwise
        cell_ = rotate(cell_, 180)
        nu_ = rotate(nu_, 180)
        img = rotate(img, 180)

    img_resized = resize(img, (shape_x, shape_y), mode="constant")
    nu_resized = resize(nu_, (shape_x, shape_y), mode="constant") * 255
    cell_resized = resize(cell_, (shape_x, shape_y), mode="constant") * 255
    pts_ori = image_warp.find_landmarks(
        nu_resized, cell_resized, n_points=cfg.N_LANDMARKS, border_points=False
    )
    
    pts_convex = (pts_avg + pts_ori) / 2
    warped1 = image_warp.warp_image(pts_ori, pts_convex, img_resized)
    warped = image_warp.warp_image(pts_convex, pts_avg, warped1)
    warped = (warped*255).astype('uint8')
    #print(warped.max())
    imwrite(save_protein_path, warped)
    if np.random.choice([True, False], p=[0.01, 0.99]):
        # Plot landmark points at morphing
        fig, ax = plt.subplots(1, 5, figsize=(15, 30))
        ax[0].imshow(nu_, alpha=0.3)
        ax[0].imshow(cell_, alpha=0.3)
        ax[0].set_title("original shape")
        ax[1].imshow(nu_resized, alpha=0.3)
        ax[1].imshow(cell_resized, alpha=0.3)
        ax[1].set_title("resized shape+protein")
        ax[2].imshow(img_resized)
        ax[2].scatter(
            pts_ori[:, 1],
            pts_ori[:, 0],
            c=np.arange(len(pts_ori)),
            cmap="Reds",
        )
        ax[2].set_title("resized protein channel")
        ax[3].imshow(warped1)
        ax[3].scatter(
            pts_convex[:, 1],
            pts_convex[:, 0],
            c=np.arange(len(pts_ori)),
            cmap="Reds",
        )
        ax[3].set_title("ori_shape to midpoint")
        ax[4].imshow(warped)
        ax[4].scatter(
            pts_avg[:, 1],
            pts_avg[:, 0],
            c=np.arange(len(pts_ori)),
            cmap="Reds",
        )
        ax[4].set_title("midpoint to avg_shape")
        fig.savefig(save_protein_path.replace(".png","_process.png"), bbox_inches="tight")
        plt.close()

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_line", type=str)
    args = parser.parse_args()
    import configs.config as cfg

    if cfg.COEF_FUNC == "fft":
        get_coef_fun = coefs.fourier_coeffs 
        inverse_func = coefs.inverse_fft 
    elif cfg.COEF_FUNC == "wavelet":
        get_coef_fun = coefs.wavelet_coefs
        inverse_func = coefs.inverse_wavelet

    cell_line = args.cell_line
    project_dir = os.path.join(os.path.dirname(cfg.PROJECT_DIR), cell_line)
    data_dir = f"{project_dir}/cell_masks"
    protein_dir = f"{project_dir}/warps"
    print(f"Saving to {protein_dir}")
    if not os.path.exists(protein_dir):
        os.makedirs(protein_dir)
    
    cellline_meta = os.path.join(project_dir, os.path.basename(cfg.META_PATH).replace(".csv", "_splitVesiclesPCP.csv"))
    mappings = pd.read_csv(cellline_meta)
    log_dir = f"{project_dir}/logs"
    fft_dir = f"{project_dir}/fftcoefs/{cfg.ALIGNMENT}"
    fft_path = os.path.join(fft_dir, f"fftcoefs_{cfg.N_COEFS}.txt")
    shift_path = os.path.join(fft_dir, f"shift_error_meta_fft{cfg.N_COEFS}.txt")
    with open(fft_path) as f:
        count = sum(1 for _ in f)
    with open(fft_path) as f_shift:
        count2 = sum(1 for _ in f_shift)

    shape_mode_path = f"{project_dir}/shapemode/{cfg.ALIGNMENT}_cell_nuclei"
    
    # Load average cell
    avg_cell = np.load(f"{shape_mode_path}/Avg_cell.npz")
    nu_centroid = [0, 0]
    ix_n = avg_cell["ix_n"]
    iy_n = avg_cell["iy_n"]
    ix_c = avg_cell["ix_c"]
    iy_c = avg_cell["iy_c"]

    pts_avg, (shape_x, shape_y) = avg_cell_landmarks(
                ix_n, iy_n, ix_c, iy_c, n_landmarks=cfg.N_LANDMARKS
            )

    completed = [os.path.basename(img_id).replace("_protein.npy","png") for img_id in glob.glob(f"{protein_dir}/*.png")]
    print(len(completed),completed[:4])
    n_processes = multiprocessing.cpu_count()
    chunk_size = count//n_processes
    print(f"processing {count - len(completed)} in {n_processes} cpu in chunk {chunk_size}")
    #mappings = mappings[mappings.image_id == '589_B4_3']
    #print(get_line(fft_path,'589_B4_3_7',mode='first').split(',')[:2])
    with open(fft_path, "r") as f:
        for l_num in tqdm(f, total=count):
            image_warping(l_num, cfg, shift_path, data_dir, protein_dir, mappings, pts_avg, shape_x, shape_y) 
        # processed_list = Parallel(n_jobs=n_processes)(
        #         delayed(image_warping)(l_num, cfg, shift_path, data_dir, protein_dir, mappings, pts_avg, shape_x, shape_y,)
        #         for l_num in tqdm(f, total=count))
        
if __name__ == "__main__":
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
