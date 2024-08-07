import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import io, models, plot
from glob import glob
from skimage import img_as_float
from skimage import exposure
from tqdm import tqdm
from natsort import natsorted

# import multiprocessing
# from joblib import Parallel, delayed


def sharpen(image):
    image = img_as_float(image)
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    return img_rescale


def adaptive_hist(image):
    img_adapteq = exposure.equalize_adapthist(image, kernel_size=110, clip_limit=0.05)
    return img_adapteq


def predict(model_path, files, plot_dir, diameter=0):
    # declare model
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)

    # use model diameter if user diameter is 0
    diameter = model.diam_labels if diameter == 0 else diameter
    print(f"object diameter: {diameter}, threshold at {diameter*0.8}")
    diameter = diameter * 0.8
    # gets image files in dir (ignoring image files ending in _masks)
    cellprob_threshold = 0
    model_name = model_path.rsplit("_")[1]
    if model_name == "nuclei":
        flow_threshold = 0.1
        channels = [2, 3]

        def read_img(f):
            w1 = io.imread(f)
            w2 = io.imread(f.replace("w1.tif", "w2.tif"))
            w3 = io.imread(f.replace("w1.tif", "w3.tif"))
            try:
                if w1.max() > 0 and w2.max() > 0 and w3.max() > 0:
                    # img = np.stack([sharpen(w1),sharpen(w2), sharpen(w3)])
                    # w23 = adaptive_hist(w2) + adaptive_hist(w3)
                    # img = np.stack([sharpen(w1),adaptive_hist(w23), sharpen(w1)])
                    img = np.stack([sharpen(w1), adaptive_hist(w2), adaptive_hist(w3)])
                else:  # fail because empty channel
                    img = []
            except:  # fail because reading error
                img = []
            return img

    elif model_name == "cyto":
        flow_threshold = 0
        channels = [2, 3]

        def read_img(f):
            w1 = io.imread(f)
            nuclei = io.imread(f.replace("w1.tif", "nucleimask.png"))
            nuclei = nuclei / nuclei.max()
            img = np.stack([np.zeros_like(w1), sharpen(w1), nuclei])
            return img

    chunk_size = 10
    n = len(files)
    with tqdm(total=n) as pbar:
        for start_ in range(0, n, chunk_size):
            end_ = min(start_ + chunk_size, n)
            images = []
            file_names = []
            for i_ in range(start_, end_):
                img = read_img(files[i_])
                if len(img) == 0:
                    with open(
                        "/data/2Dshapespace/S-BIAD34/resegmentation/failed_imgs_channelvalue0.txt",
                        "a",
                    ) as f:
                        print(f'Failed: {files[i_].split("/")[-2:]}')
                        f.write(files[i_])
                else:
                    images += [img]
                    file_names += [files[i_]]
            # run model on <chunk_size> images
            masks, flows, styles = model.eval(
                images,
                channels=channels,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
            )

            if True:
                # Random QC

                if True:
                    i = np.random.choice(len(images))
                    # for i in range(1, len(images)):
                    fig = plt.figure(figsize=(40, 10), facecolor="black")
                    name = os.path.basename(file_names[i]).replace("_w1.tif", ".png")
                    name = "_".join([file_names[i].split("/")[-2], name])
                    print(name)
                    img = images[i].copy()
                    if img.shape[0] != len(channels):
                        tmp = []
                        for ch in channels:
                            tmp += [img[ch - 1]]
                        img = np.stack(tmp) * 0.3
                    plot.show_segmentation(
                        fig,
                        img,
                        masks[i],
                        flows[i][0],
                        channels=channels,
                        file_name=None,
                    )
                    fig.savefig(f"{plot_dir}/{name}")
                    plt.close()
                    # io.imsave(f'{plot_dir}/{name[:-4]}_{model_name}mask.png',masks[i])
            for m, f in zip(masks, file_names):
                last_pattern = f.split("_")[-1]
                io.imsave(f.replace(last_pattern, f"{model_name}mask.png"), m)
            pbar.update(end_ - start_)


if __name__ == "__main__":
    base_dir = "/data/2Dshapespace/S-BIAD34"
    if False:
        files = natsorted(glob(f"{base_dir}/resegmentation/train/*w1.tif"))
        files = [
            f
            for f in files
            if not os.path.exists(f.replace("w1.tif", "nucleimask.png"))
        ]
        print(f"==========> Segmenting nucleus")
        os.makedirs(f"{base_dir}/resegmentation/QCs/nuclei", exist_ok=True)
        predict(
            model_path=f"{base_dir}/resegmentation/models/S-BIAD34_nuclei",
            files=files,
            plot_dir=f"{base_dir}/resegmentation/QCs/nuclei",
        )
    # standardize extension to .tif
    files = natsorted(glob(f"{base_dir}/Files/*/*.TIF"))
    for f in files:
        os.rename(f, f.replace(".TIF", ".tif"))

    if True:
        files_finished = natsorted(glob(f"{base_dir}/Files/*/*nucleimask.png"))
        files_finished = [f.replace("nucleimask.png", "w1.tif") for f in files_finished]
        print(f"Found {len(files_finished)} FOVs with nucleimasks.png done")
        files = natsorted(glob(f"{base_dir}/Files/*/*w1.tif"))
        files = [f for f in files if f not in files_finished]
        print(f"========== Segmenting {len(files)} fovs ==========")
        print(f"==========> Segmenting nucleus")
        os.makedirs(f"{base_dir}/resegmentation/QCs/nuclei", exist_ok=True)
        predict(
            model_path=f"{base_dir}/resegmentation/models/S-BIAD34_nuclei",
            files=files,
            plot_dir=f"{base_dir}/resegmentation/QCs/nuclei",
        )

    if True:
        files_finished = natsorted(glob(f"{base_dir}/Files/*/*cytomask.png"))
        files_finished = [f.replace("cytomask.png", "w1.tif") for f in files_finished]
        files = natsorted(glob(f"{base_dir}/Files/*/*nucleimask.png"))
        files = [f.replace("nucleimask.png", "w1.tif") for f in files]
        files = [f for f in files if f not in files_finished]
        print(f"========== Segmenting {len(files)} fovs ==========")
        print(f"==========> Segmenting cells")
        os.makedirs(f"{base_dir}/resegmentation/QCs/cell", exist_ok=True)
        predict(
            model_path=f"{base_dir}/resegmentation/models/S-BIAD34_cyto",
            files=files,
            plot_dir=f"{base_dir}/resegmentation/QCs/cell",
        )
