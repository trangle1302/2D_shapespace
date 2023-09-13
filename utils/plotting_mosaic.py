import os
import numpy as np
import torch
from torchvision.utils import make_grid


def make_mosaic():
    cdts = np.linspace(0.0, -3.0, 20) # so the grid with go top to bottom,
    gmnns = np.linspace(-3.0, 0.0, 20) # left to right
    nrow = len(gmnns)
    images = []
    blanks = 0
    for cdt in cdts:
        for gmnn in gmnns:
            # find the image closest to this point in the intensity space
            point = torch.tensor([gmnn, cdt])
            distances = torch.sum((FUCCI_log_mean_intensities_nonzero - point) ** 2, dim=1)
            closest = torch.argmin(distances)
            # if the closest is within 0.1, then we can use it, else just add a blank image
            if distances[closest] > ((0.5 * (gmnns[1] - gmnns[0])) ** 2 + (0.5 * (cdts[1] - cdts[0])) ** 2):
                images.append(torch.zeros_like(dataset[closest]))
                blanks += 1
            else:
                images.append(dataset[closest])

    print(len(images))
    print(blanks)
    print(images[0].shape)

    grid = make_grid(images, nrow=nrow)
    print(grid.shape)