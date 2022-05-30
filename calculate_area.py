# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 12:06:12 2022

@author: trang.le
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
  18: 'Negative',
  19:'Multi-Location',
}

COLORS = [
    '#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5',
    '#2196f3', '#03a9f4', '#00bcd4', '#009688', '#4caf50',
    '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800',
    '#ff5722', '#795548', '#9e9e9e', '#607d8b', '#dddddd',
    '#212121', '#ff9e80', '#ff6d00', '#ffff00', '#76ff03',
    '#00e676', '#64ffda', '#18ffff',
]

df = pd.read_csv("./Downloads/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d.csv")
df.shape
X = df[['x', 'y', 'z']].to_numpy()
sub_df = df
show_multi = True
num_classes = 20 if show_multi else 19
fig, ax = plt.subplots(figsize=(32, 16))
for i in range(num_classes):
    label = LABEL_TO_ALIAS[i]
    if label in ['Negative','VesiclesPCP','Cytosol']:
        continue
    idx = np.where(sub_df['target']==label)[0]
    x = X[idx, 0]
    y = X[idx, 1]
    #print(label, sub_df['Label'][idx])
    if label=="Multi-Location":
        plt.scatter(x, y, c=COLORS[i],label=LABEL_TO_ALIAS[i], s=8, alpha=0.05)
    else:
        plt.scatter(x, y, c=COLORS[i],label=LABEL_TO_ALIAS[i], s=8)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width* 0.8, box.height])
ax.legend(loc='upper right', fontsize=24, bbox_to_anchor=(1.24, 1.01), ncol=1, markerscale=6)
