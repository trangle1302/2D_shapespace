import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import utils, io, models, plot
from glob import glob
import tqdm

def predict(model_path, files):
    # gets image files in dir (ignoring image files ending in _masks)
    files = io.get_image_files(dir, '_masks')
    print(files)
    images = [io.imread(f) for f in files]
    flow_threshold = 0.1
    cellprob_threshold = 0
    if model_path.rsplit('_')[1] == 'nuclei':
        channels = [2,3]
    elif model_path.rsplit('_')[1] == 'cyto':
        channels = [1,2]
    # declare model
    model = models.CellposeModel(gpu=True, 
                                pretrained_model=model_path)

    # use model diameter if user diameter is 0
    diameter = model.diam_labels if diameter==0 else diameter

    # run model on test images
    masks, flows, styles = model.eval(images, 
                                    channels=channels,
                                    diameter=diameter,
                                    flow_threshold=flow_threshold,
                                    cellprob_threshold=cellprob_threshold
                                    )
    
    # Random QC
    fig = plt.figure(figsize=(40,10))
    i = np.random.choice(len(images))
    img = images[i].copy() * 0.5
    plot.show_segmentation(fig, img, masks[i], flows[i][0], channels=channels, file_name=None)

if __name__ == "__main__": 
    base_dir = '/data/2Dshapespace/S-BIAD34'
    files = glob.glob(f'{base_dir}/Files/*/*w1.tif')
    print(f'========== Segmenting {len(files)} fovs ==========')
    print(f'==========> Segmenting nucleus')
    predict(model_path = f'{base_dir}/resegments/models/S-BIAD34_nuclei', )
    print(f'==========> Segmenting cells')
    predict(model_path = f'{base_dir}/resegments/models/S-BIAD34_cyto', )
