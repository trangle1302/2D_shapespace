import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
from glob import glob
from skimage import img_as_float
from skimage import exposure
from natsort import natsorted

def sharpen(image):
    image = img_as_float(image)
    p5, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p5, p98))
    return img_rescale

def train(train_files, test_files, save_dir, initial_model='nuclei'):
    use_GPU=True
    n_epochs = 200
    learning_rate = 0.05
    weight_decay = 0.001
    if initial_model == 'nuclei':
        model_name = 'S-BIAD34_nuclei'
        channels = [2, 3]
            
        train_data = []
        train_labels = []
        for k, f in enumerate(train_files):
            w1 = io.imread(f)
            nuclei = io.imread(f.replace('w1.tif','nucleimask.png'))
            img = np.stack([sharpen(w1),nuclei, np.zeros_like(w1)])
            train_data += [img]
            train_labels += [io.imread(f.replace('w1.tif','nucleimask.png'))]

        test_data = []
        test_labels = []
        for k, f in enumerate(test_files):
            w1 = io.imread(f)
            nuclei = io.imread(f.replace('w1.tif','nucleimask.png'))
            img = np.stack([sharpen(w1), nuclei, np.zeros_like(w1)])
            test_data += [img]
            test_labels += [io.imread(test_seg[k].replace('w1.tif','nucleimask.png'))]


    elif initial_model == 'cyto':
        model_name = 'S-BIAD34_cyto'
        channels = [1, 2]
        train_data = []
        train_labels = []
        for k, f in enumerate(train_files):
            w1 = io.imread(f)
            nuclei = io.imread(f.replace('w1.tif','nucleimask.png'))
            img = np.stack([sharpen(w1),nuclei, np.zeros_like(w1)])
            train_data += [img]
            train_labels += [io.imread(f.replace('w1.tif','cellmask.png'))]

        test_data = []
        test_labels = []
        for k, f in enumerate(test_files):
            w1 = io.imread(f)
            nuclei = io.imread(f.replace('w1.tif','nucleimask.png'))
            img = np.stack([sharpen(w1), nuclei, np.zeros_like(w1)])
            test_data += [img]
            test_labels += [io.imread(test_seg[k].replace('w1.tif','cellmask.png'))]


    # start logger (to see training across epochs)
    logger = io.logger_setup()

    # DEFINE CELLPOSE MODEL (without size model)
    model = models.CellposeModel(gpu=use_GPU, model_type=initial_model)

    new_model_path = model.train(train_data, train_labels, 
                                test_data=[],
                                test_labels=[],
                                channels=channels, 
                                save_path=save_dir, 
                                n_epochs=n_epochs,
                                learning_rate=learning_rate, 
                                weight_decay=weight_decay, 
                                nimg_per_epoch=2,
                                model_name=model_name)

    # diameter of labels in training images
    diam_labels = model.diam_labels.copy()
    masks = model.eval(test_data, 
                   channels=[0, 1, 2],
                   diameter=diam_labels)[0]
    save_path = f"{save_dir}/test_predictions.png"
    plot(test_data, test_labels, masks, save_path)

def plot(data, groundtruths, predicted_masks, save_path):   
    try:
        n = data.shape[0] #data is in np.array, so batch is at index 0
    except:
        n = len(data) #data is a list
    plt.figure(figsize=(12,8), dpi=150)
    for k,im in enumerate(data):
        img = im.copy()
        plt.subplot(3, n, k+1)
        img = np.vstack((img, np.zeros_like(img)[:1]))
        img = img.transpose(1,2,0)
        plt.imshow(img)
        plt.axis('off')
        if k==0:
            plt.title('image')

        plt.subplot(3, n, n + k+1)
        plt.imshow(predicted_masks[k])
        plt.axis('off')
        if k==0:
            plt.title('predicted labels')

        plt.subplot(3, n, 2*n + k+1)
        plt.imshow(groundtruths[k])
        plt.axis('off')
        if k==0:
            plt.title('true labels')
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__": 
    #base_dir = '/content/gdrive/MyDrive/Files'
    base_dir = '/data/2Dshapespace/S-BIAD34/resegmentation'
    train_files = glob(f'{base_dir}/train/*_w1.tif')
    test_files = glob(f'{base_dir}/test/*_w1.tif')

    train(train_files, test_files,save_dir=base_dir, initial_model='nuclei')
    train(train_files, test_files,save_dir=base_dir, initial_model='cyto')