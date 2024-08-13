import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import io, models, plot
from glob import glob
from skimage import img_as_float
from skimage import exposure
from tqdm import tqdm
from natsort import natsorted
import concurrent.futures
import time


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
    diameter = diameter * 1.5
    # gets image files in dir (ignoring image files ending in _masks)
    cellprob_threshold = 0
    model_name = model_path.rsplit("_")[1]
    if model_name == "nuclei":
        flow_threshold = 0
        channels = [2, 3]

        def read_img(f):
            # w4 = io.imread(f)
            w0 = io.imread(f.replace("C4.tif", "C0.tif"))
            # w1 = io.imread(f.replace("C4.tif", "C1.tif"))
            # print(f'read {w1.max(), w0.max(), w4.max()}')
            try:
                # if w1.max() > 0 and w0.max() > 0 and w4.max() > 0:
                if w0.max() > 0:
                    w0 = sharpen(w0)  # rescale intensity
                    img = np.stack(
                        [adaptive_hist(w0), adaptive_hist(w0), adaptive_hist(w0)]
                    )
                else:  # fail because empty channel
                    img = []
            except:  # fail because reading error
                img = []
            return img, f

    elif model_name == "cyto":
        flow_threshold = 0
        channels = [2, 3]

        def read_img(f):
            w4 = io.imread(f)  # C4 microtubules
            w1 = io.imread(f.replace("C4.tif", "C1.tif"))  # C1 ER
            w4 = w1 + w4
            nuclei = io.imread(f.replace("C4.tif", "nucleimask.png"))
            nuclei = nuclei / nuclei.max()
            img = np.stack([sharpen(w1), sharpen(w4), nuclei])
            return img, f

    chunk_size = 64
    n = len(files)
    with tqdm(total=n) as pbar:
        for start_ in range(0, n, chunk_size):
            end_ = min(start_ + chunk_size, n)
            images = []
            file_names = []
            s = time.time()
            # concurrent futures increase reading images by 3x
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(read_img, files[i_]) for i_ in range(start_, end_)
                ]
                for future in concurrent.futures.as_completed(futures):
                    img, file_name = future.result()
                    if len(img) == 0:
                        with open(
                            "/scratch/users/tle1302/2Dshapespace/B2AI/failed_imgs_channelvalue0.txt",
                            "a",
                        ) as f:
                            print(f'Failed: {file_name.split("/")[-3:]}')
                            f.write(file_name + "\n")
                    else:
                        images.append(img)
                        file_names.append(file_name)

            print(f"Loading {chunk_size} image took {(time.time() - s)/60} min")
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
                    if True:
                        fig = plt.figure(figsize=(40, 10), facecolor="black")
                        name = os.path.basename(file_names[i]).replace(".tif", ".png")
                        name = "_".join(
                            [
                                file_names[i].split("/")[-4],
                                file_names[i].split("/")[-3],
                                file_names[i].split("/")[-2],
                                name,
                            ]
                        )
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
                        print(f"savng : {plot_dir}/{name}")
                        fig.savefig(f"{plot_dir}/{name}")
                        plt.close()
                    # io.imsave(f'{plot_dir}/{name[:-4]}_{model_name}mask.png',masks[i])
            for m, f in zip(masks, file_names):
                io.imsave(f.replace("C4.tif", f"{model_name}mask.png"), m)
            pbar.update(end_ - start_)


def delete_empty_lowcount(folders):
    for f in folders:
        for ch in ["C0.tif", "C1.tif", "C2.tif", "C3.tif", "C4.tif"]:
            img = io.imread(f"{f}/{ch}")
            if img.max() < 100:
                print(f"Removing {f}/{ch}")
                os.remove(f"{f}/{ch}")


if __name__ == "__main__":
    base_dir = "/scratch/users/tle1302/2Dshapespace/B2AI/MDA-MB-468/Tiffs/B2AI-2023-1/"

    if False:
        folders = natsorted(glob(f"{base_dir}/*/*/*"))
        delete_empty_lowcount(folders)

    if True:
        files_finished = natsorted(glob(f"{base_dir}/*/*/*/*nucleimask.png"))
        files_finished = [f.replace("nucleimask.png", "C4.tif") for f in files_finished]
        print(f"Found {len(files_finished)} FOVs with nucleimasks.png done")
        files = natsorted(glob(f"{base_dir}/*/*/*/C4.tif"))  # w4 is microtubules
        files = [f for f in files if f not in files_finished]
        print(f"========== Segmenting {len(files)} fovs ==========")

        print(f"==========> Segmenting nucleus")
        os.makedirs(f"/scratch/users/tle1302/2Dshapespace/QCs/nuclei", exist_ok=True)
        predict(
            model_path=f"/scratch/users/tle1302/2Dshapespace/models/S-BIAD34_nuclei",
            files=files,
            plot_dir=f"/scratch/users/tle1302/2Dshapespace/QCs/nuclei",
        )

    if True:
        # files_finished = natsorted(glob(f"{base_dir}/*/*/*/*cytomask.png"))
        # files_finished = [f.replace('nucleimask.png','C4.tif') for f in files_finished]
        files = natsorted(glob(f"{base_dir}/*/*/z01/nucleimask.png"))
        files = [f.replace("nucleimask.png", "C4.tif") for f in files]
        # files = [f for f in files if f not in files_finished]
        print(f"========== Segmenting {len(files)} fovs ==========")
        print(f"==========> Segmenting cells")
        os.makedirs(f"/scratch/users/tle1302/2Dshapespace/QCs/cell", exist_ok=True)
        predict(
            model_path=f"/scratch/users/tle1302/2Dshapespace/models/S-BIAD34_cyto",
            files=files,
            plot_dir=f"/scratch/users/tle1302/2Dshapespace/QCs/cell",
        )
