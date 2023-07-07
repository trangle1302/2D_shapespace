import os
import sys
sys.path.append("..")
import numpy as np
from utils import helpers
from utils.helpers import grep
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, rotate
from skimage.transform import resize
# from warps import image_warp
from warps import image_warp_new as image_warp
import json
import pandas as pd
from tqdm import tqdm
import time
import gc
from collections import Counter
import argparse


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
        ix_n, iy_n = helpers.equidistance(ix_n, iy_n, n_points=n_landmarks)
        ix_c, iy_c = helpers.equidistance(ix_c, iy_c, n_points=n_landmarks * 2)
    nu_contour = np.stack([ix_n, iy_n]).T
    cell_contour = np.stack([ix_c, iy_c]).T
    # print(nu_contour.shape, cell_contour.shape)

    pts_avg = np.vstack(
        [
            np.asarray(nu_centroid),
            helpers.realign_contour_startpoint(nu_contour),
            helpers.realign_contour_startpoint(cell_contour),
        ]
    )
    # print(pts_avg.max(), pts_avg.min(), cell_contour[:,0].max(), cell_contour[:,1].max())
    shape_x, shape_y = (
        np.round(cell_contour[:, 0].max()).astype("int"),
        np.round(cell_contour[:, 1].max()).astype("int"),
    )

    return pts_avg, (shape_x, shape_y)


def main():
    s = time.time()
    import configs.config as cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_bins", nargs="+", help="bin to investigate", type=int)
    parser.add_argument("--pc", help="", type=str)
    args = parser.parse_args()
    bin_ = args.merged_bins
    PC = args.pc
    if cfg.SERVER == "callisto":
        from imageio import imread, imwrite
    elif cfg.SERVER == "sherlock":
        from imageio.v2 import imread, imwrite 
    shape_mode_path = f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}"
    fft_dir = f"{cfg.PROJECT_DIR}/fftcoefs/{cfg.ALIGNMENT}"
    data_dir = f"{cfg.PROJECT_DIR}/cell_masks2"
    save_dir = f"{cfg.PROJECT_DIR}/morphed_protein_avg"
    plot_dir = f"{cfg.PROJECT_DIR}/morphed_protein_avg_plots"
    n_landmarks = 32  # number of landmark points frgs='+',or each ring, so final n_points to compute dx, dy will be 2*n_landmarks+1
    # print(save_dir, plot_dir)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Loading cell assignation into PC bins
    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json", "r")
    cells_assigned = json.load(f)
    print(cells_assigned.keys())
    # mappings = pd.read_csv(f"{project_dir}/experimentB-processed.txt", sep="\t")
    # print(f"...Found {len(mappings['Antibody id'].unique())} antibodies")

    pro_count = {}
    for b in np.arange(7):
        pc_cells = cells_assigned[PC][b]
        antibodies = [c.split("/")[-2] for c in pc_cells]
        cells_per_ab = Counter(antibodies)
        pro_count[f"bin{b}"] = cells_per_ab
        print(len(pc_cells), len(cells_per_ab.keys()))

    df = pd.DataFrame(pro_count)
    df["total"] = df.sum(axis=1)
    # print(df.sort_values(by=['total']))
    idx_keep = [all(r.values >= 5) for _, r in df.iterrows()]
    ab_keep = df.index.values[idx_keep].tolist()
    print(f"Keeping {sum(idx_keep)} ab with >=5 cells/bin")
    avg_cell_per_bin = np.load(f"{shape_mode_path}/shapevar_{PC}_cell_nuclei.npz")

    shift_path = f"{fft_dir}/shift_error_meta_fft128.txt"

    print(f"Processing {bin_} of {PC}")
    # created a folder where avg protein for each bin is saved
    os.makedirs(f"{save_dir}/{PC}", exist_ok=True)

    pc_cells = cells_assigned[PC]
    if True:
        if len(bin_) == 1:
            n_coef = len(avg_cell_per_bin["nuc"][0]) // 2
            ix_n = avg_cell_per_bin["nuc"][bin_[0]][:n_coef]
            iy_n = avg_cell_per_bin["nuc"][bin_[0]][n_coef:]
            ix_c = avg_cell_per_bin["mem"][bin_[0]][:n_coef]
            iy_c = avg_cell_per_bin["mem"][bin_[0]][n_coef:]
            pts_avg, (shape_x, shape_y) = avg_cell_landmarks(
                ix_n, iy_n, ix_c, iy_c, n_landmarks=n_landmarks
            )

        ls = [pc_cells[b] for b in bin_]
        ls = helpers.flatten_list(ls)
        # print("examples of antibodies", ab_keep[:5])
        # ab_keep = ["HPA030782","HPA050556","HPA051349","HPA036914","HPA040748"]
        for ab_id in ab_keep:
            save_path = f"{save_dir}/{PC}/{ab_id}_bin{bin_[0]}.png"
            if os.path.exists(save_path):
                continue
            print(f"Preparing for {PC}/{ab_id}_bin{bin_[0]}.png")
            print(len(ls), len([f for f in ls if f.__contains__(ab_id)]))
            # 1 empty avg_img (initialization) for each protein_pc_bin combination
            # avg_img = np.zeros((shape_x+2, shape_y+2), dtype='float64')
            avg_img = np.zeros((shape_x, shape_y), dtype="float64")
            if not os.path.isdir(f"{plot_dir}/{PC}/{ab_id}"):
                os.makedirs(f"{plot_dir}/{PC}/{ab_id}")
            ls_ = [f for f in ls if f.__contains__(ab_id)]
            ls_ = [os.path.basename(l).replace(".npy", "") for l in ls_]
            print(f"There are {len(ls_)} proteins for this {ab_id}_bin{bin_[0]} ")
            for img_id in tqdm(ls_, desc=f"{PC}_bin{bin_[0]}_{ab_id}", total=len(ls_)):
                line_ = grep(img_id+".npy", shift_path)                            
                if line_ == []:
                    print(f"{img_id} not found")
                    return 
                vals = line_[0].strip().split(";") 
                theta = float(vals[1])
                shift_c = (
                    float(vals[2].split(",")[0].strip("(")),
                    (float(vals[2].split(",")[1].strip(")"))),
                )
                #print(vals, f"{data_dir}/{ab_id}/{img_id}_protein.png")
                cell_shape = np.load(f"{data_dir}/{ab_id}/{img_id}.npy")
                img_ori = imread(f"{data_dir}/{ab_id}/{img_id}_protein.png")
                if img_ori.dtype == "uint16":
                    img_ori = (img_ori / 256).astype(np.uint8)
                #print(f"Image value max {img_ori.max()}, image dtype: {img_ori.dtype}")
                img = rotate(img_ori, theta)
                nu_ = rotate(cell_shape[1, :, :], theta)
                cell_ = rotate(cell_shape[0, :, :], theta)

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
                # print(f"rotated img max: {img.max()}, resized img max: {img_resized.max()}")
                # print(f"rotated nu max: {nu_.max()}, resized nu max: {nu_resized.max()}, rotated cell max: {cell_.max()}, resized cell max: {cell_resized.max()}")
                pts_ori = image_warp.find_landmarks(
                    nu_resized, cell_resized, n_points=n_landmarks, border_points=False
                )
                #print(img_resized.shape, avg_img.shape, (shape_x, shape_y))
                pts_convex = (pts_avg + pts_ori) / 2
                warped1 = image_warp.warp_image(
                    pts_ori, pts_convex, img_resized
                )  # , plot=False, save_dir="")
                warped = image_warp.warp_image(
                    pts_convex, pts_avg, warped1
                )  # , plot=False, save_dir="")
                # imwrite(f"{save_dir}/{PC}/{org}/{img_id}.png", (warped*255).astype(np.uint8))

                # adding weighed contribution of this image
                # print("Accumulated: ", avg_img.max(), avg_img.dtype, "Addition: ", warped.max(), warped.dtype)
                avg_img += warped / len(ls_)
                if (
                    True
                ):  # np.random.choice([True,False], p=[0.01,0.99]) or ab_id in ["HPA001644", "HPA050627"]:
                    # if ab_id in ["HPA049341","HPA061027","HPA060948","HPA063464","HPA065938","HPA040923","HPA032080","HPA030741"] and np.random.choice([True,False], p=[0.1,0.9]):
                    # Plot landmark points at morphing
                    fig, ax = plt.subplots(1, 6, figsize=(15, 35))
                    ax[0].imshow(cell_shape[1, :, :], alpha=0.3)
                    ax[0].imshow(cell_shape[0, :, :], alpha=0.3)
                    ax[0].set_title("original shape")
                    ax[1].imshow(img_ori)
                    ax[1].set_title("original protein intensity")
                    ax[2].imshow(nu_resized, alpha=0.3)
                    ax[2].imshow(cell_resized, alpha=0.3)
                    ax[2].set_title(
                        f"resized shape+protein \n theta = {np.round(theta,1)}°"
                    )
                    ax[3].imshow(img_resized)
                    ax[3].scatter(
                        pts_ori[:, 1],
                        pts_ori[:, 0],
                        c=np.arange(len(pts_ori)),
                        cmap="Reds",
                        alpha=0.5,
                    )
                    ax[3].set_title("resized protein")
                    ax[4].imshow(warped1)
                    ax[4].scatter(
                        pts_convex[:, 1],
                        pts_convex[:, 0],
                        c=np.arange(len(pts_ori)),
                        cmap="Reds",
                        alpha=0.5,
                    )
                    ax[4].set_title("ori_shape to midpoint")
                    ax[5].imshow(warped)
                    ax[5].scatter(
                        pts_avg[:, 1],
                        pts_avg[:, 0],
                        c=np.arange(len(pts_ori)),
                        cmap="Reds",
                        alpha=0.5,
                    )
                    ax[5].set_title("midpoint to avg_shape")
                    fig.savefig(
                        f"{plot_dir}/{PC}/{ab_id}/{img_id}.png", bbox_inches="tight"
                    )
                    plt.close()
            print(
                "Accumulated: ",
                avg_img.max(),
                avg_img.dtype,
                "Addition: ",
                warped.max(),
                warped.dtype,
            )
            print(f"Saving to {save_path}")
            imwrite(save_path, (avg_img * 255).astype(np.uint8))
            gc.collect()
    print(f"Time elapsed: {(time.time() - s)/3600} h.")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except:
        sys.exit(1)
