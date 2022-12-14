# Pipeline to process shape modes

pilot: U2OS cell lines
small subsets (private images, manual segmentation and annotations): cells (1776 images)
HPA (public images, automatic segmentation): 297108 cells (23272 images)

Steps for the pipelines:
## s0 - segmentation
Either manual segmentation, or segmentation by any DL model (in this case HPACellSegmentator).

## s1 - process image masks of multiple cells to single cell masks of cell and nucleus, and into .npy

Folder: [segmentation](https://github.com/trangle1302/2D_shapespace/tree/master/segmentation) 

Removing cells where nucleus touching the borders. Cells where cell segmentation touching the bordered are still kept (maybe do a percentage rules to remove them in the future).
```sh
python s1_get_single_cell_shapes.py
```

## s2 - get FFT coeficients for individua cell and nucleus shapes

Folder: [coefficients](https://github.com/trangle1302/2D_shapespace/tree/master/coefficients) 

- Alignment and center: major axis, nuclei-cell centroid vector, major axis + nuclei centroid (mass) alignment
- Calculate FFT of x,y of the nucleus and cell segmentation (equally spaced sample along the shapes): fast fourier coefficients, elliptical fourier discriptors, wavelet
- Save result of multiprocessing pool

```sh
python s2_calculate_fft.py
```

## s3 - Calculate shape modes & map of single-organelle protein

Folder: [shapemodes](https://github.com/trangle1302/2D_shapespace/tree/master/shapemodes) 

Fit and transform PCA, calculate shapemodes (n_PCs with xx% variance) based on coefficients produced from s2.
```sh
python s3_calculate_shapemodes.py
```

## s4 - Protein parameterization: Intepolate concentric rings in green channels and shape modes

Folder: [warp](https://github.com/trangle1302/2D_shapespace/tree/master/warp) 

Protein parameterization based on concentric rings from nucleus centroid - nucleus membrane - cell membrane. Final shape for all proteins: (n_rings, n_points)
```sh
python s4_concentric_rings_intensity.py
```
OR
Protein morphing on to shape based on [thin-plate splines](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=24792) given landmarks: nucleus centroid, 32p in nucleus membrane, 32p cell membrane. Final shape for all proteins = shape of the average cell in that shapemode (bin).
```sh
python s4_protein_image_warp.py
```

## s5 - Organelle distribution and relation with each other

```sh
python s5_organelle_heatmappy
```


# Mislanchelous shell scripts:
- Count number of lines in a text file:

- Replace old_text with new_text in a big file 
```sh
time sed -i 's|old_text|new_text|g' filename.xxx
time sed -i 's|/data/2D|\n/data/2D|g' shift_error_meta_fft128_2.txt
```