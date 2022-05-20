# Pipeline to process shape modes

pilot: U2OS cell lines
small subsets (private images, manual segmentation and annotations): cells (1776 images)
HPA (public images, automatic segmentation): 297108 cells (23272 images)

Steps for the pipelines:
## s0 - segmentation
Either manual segmentation, or segmentation by any DL model (in this case HPACellSegmentator).

## s1 - process image masks of multiple cells to single cell masks of cell and nucleus, and into .npy
Removing cells where nucleus touching the borders. Cells where cell segmentation touching the bordered are still kept (maybe do a percentage rules to remove them in the future).

## s2 - get FFT coeficients for individua cell and nucleus shapes
Alignment and center
Calculate FFT of x,y of the nucleus and cell segmentation (equally spaced sample along the shapes)
Save result of multiprocessing pool

## s3 - Calculate shape modes


## s4 - Intepolate concentric rings in green channels and shape modes, 
