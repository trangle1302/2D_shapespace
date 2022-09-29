import os
import numpy as np
from PIL import Image, ImageSequence
from matplotlib import pyplot as plt

LABEL_TO_ALIAS = {
  0: 'Nucleoplasm',
  1: 'NuclearM',
  2: 'Nucleoli',
  3: 'NucleoliFC',
  4: 'NuclearS',
  5: 'NuclearB',
  6: 'EndoplasmicR',
  7: 'GolgiA',
  8: 'IntermediateF',
  9: 'ActinF',
  10: 'Microtubules',
  11: 'MitoticS',
  12: 'Centrosome',
  13: 'PlasmaM',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'VesiclesPCP',
  19: 'Negative',
  19:'Multi-Location',
}

all_locations = dict((v, k) for k,v in LABEL_TO_ALIAS.items())

def open_gif(gif_path):
    animated_gif = Image.open(gif_path)#f"{organelle_dir}/{org}_PC1.gif")
    frames = [f for f in ImageSequence.Iterator(animated_gif)]
    frames = []
    for frame in ImageSequence.Iterator(animated_gif):
        fr = frame.copy()
        fr.past()
def main():
    shape_var_dir = "./Desktop/shapemode/organelle"
    organelle_dir = "./Desktop/shapemode/organelle"
    avg_cell = plt.imread(ac)
    for org in all_locations.keys()[:-1]:
        intensity = np.load(f"{organelle_dir}/{org}_PC1_intensity.npy")
        intensity.shape
        plt.imshow(intensity)
    
    
if __name__ == '__main__':
    main()