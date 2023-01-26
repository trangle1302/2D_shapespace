import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import io, models, plot
from glob import glob
from skimage import img_as_float
from skimage import exposure
from tqdm import tqdm
#import multiprocessing
#from joblib import Parallel, delayed

def sharpen(image):
    image = img_as_float(image)
    p5, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p5, p98))
    return img_rescale

def predict(model_path, files, plot_dir):
    
    # declare model
    model = models.CellposeModel(gpu=True, 
                                pretrained_model=model_path)

    # use model diameter if user diameter is 0
    diameter = model.diam_labels if diameter==0 else diameter

    # gets image files in dir (ignoring image files ending in _masks)
    flow_threshold = 0.1
    cellprob_threshold = 0
    if model_path.rsplit('_')[1] == 'nuclei':
        channels = [2,3]
        def read_img(f):
            w1 = io.imread(f)
            w2 = io.imread(f.replace('w1.tif','w2.tif'))
            w3 = io.imread(f.replace('w1.tif','w3.tif'))
            img = np.stack([sharpen(w1),sharpen(w2), sharpen(w3)])
            return img
    elif model_path.rsplit('_')[1] == 'cyto':
        channels = [1,2]
        def read_img(f):
            w1 = io.imread(f)
            nuclei = io.imread(f.replace('w1.tif','nucleimask.png'))
            img = np.stack([sharpen(w1),nuclei, np.zeros_like(w1)])
            return img

    chunk_size = 10
    n = len(files)
    with tqdm(total=n) as pbar:
        for start_ in range(0, n, chunk_size):
            end_ = min(start_ + chunk_size, n)
            images = []
            for i_ in range(start_, end_):
                images += [read_img(files[i_])]

            # run model on <chunk_size> images
            masks, flows, styles = model.eval(images, 
                                            channels=channels,
                                            diameter=diameter,
                                            flow_threshold=flow_threshold,
                                            cellprob_threshold=cellprob_threshold
                                            )
            
            # Random QC
            fig = plt.figure(figsize=(40,10))
            i = np.random.choice(len(images))
            name = os.path.basename(files[i]).replace("_w1.tif",".png")
            img = images[i].copy() * 0.5
            plot.show_segmentation(fig, img, masks[i], flows[i][0], channels=channels, file_name=None)
            fig.savefig(f'{plot_dir}/{name}')
            pbar.update(end_ - start_)

if __name__ == "__main__": 
    base_dir = '/data/2Dshapespace/S-BIAD34'
    files = glob.glob(f'{base_dir}/Files/*/*w1.tif')
    print(f'========== Segmenting {len(files)} fovs ==========')

    print(f'==========> Segmenting nucleus')
    os.makedirs(f'{base_dir}/resegment/QCs/nuclei', exist_ok=True)
    predict(model_path = f'{base_dir}/resegments/models/S-BIAD34_nuclei', files = files, plot_dir = f'{base_dir}/resegment/QCs/nuclei')
    
    print(f'==========> Segmenting cells')
    os.makedirs(f'{base_dir}/resegment/QCs/cell', exist_ok=True)
    predict(model_path = f'{base_dir}/resegments/models/S-BIAD34_cyto', files = files, plot_dir = f'{base_dir}/resegment/QCs/cell')
