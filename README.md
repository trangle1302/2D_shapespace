# 2D Shapespace

This repository contains the code and analysis for our paper, "Cell shapes decode molecular phenotypes in image-based spatial proteomics" (Le et al., 2025).

We introduce a computational framework called Shapespace, which study interpretable shape variations, and maps single-cell protein localization and pathway activity onto a common coordinate system defined by cell and nuclear morphology. This enables robust, interpretable analysis across morphological variation, conditions, and perturbations.

[notebooks/](notebooks) and [analysis/](analysis) – Python script or Jupyter notebooks for reproducing figures, performing downstream analysis, and exploring results. For step-by-step details, please refer to the Methods section of the manuscript.

This repo is not maintained.

## Installation

Clone this repository and set up a Python environment:

```bash
git clone https://github.com/CellProfiling/2D_shapespace.git
cd 2D_shapespace
```
We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [venv](https://docs.python.org/3/library/venv.html) to create an isolated environment:

```bash
# With conda
conda create -n shapespace python=3.11
conda activate shapespace

# Or with venv
python -m venv shapespace
source shapespace/bin/activate   # Linux/macOS
```

Install the repository as a Python package in editable mode:

```bash
pip install -e .
```
## Usage
Before running the analysis, you need to:

- Edit [config.py](configs/config.py) to point to your project directory, alignment mode, cell line, mapping method etc.

- Follow the workflow steps, which are described in detail in the manuscript and summarized below in the [Shapespace construction](## Shapespace construction) section.

We also provide a test dataset (single-cell crops and masks) which allows you to quickly test shape parameterization, constructing shapespace and map protein intensity to the average cell shape.

```bash
wget https://ell-vault.stanford.edu/dav/trangle/www/K-562.zip
unzip K-562.zip -d K-562

python -m coefficients.s2_calculate_fft
python -m shapemodes.s3_calculate_shapemodes
python -m warp.s4_concentric_rings_intensity # check cfg.N_ISOS and cfg.LANDMARKS
python -m warp.s4_protein_image_warp # check cfg.LANDMARKS
```
For large datasets or when analyzing multiple cell lines, consider using a workflow manager such as **Snakemake**, or submitting separate jobs to a compute cluster using **SLURM**. Example workflow files and job scripts can be found inside each folder. 

## Shapespace construction

Steps for the pipeline:
### s0 - segmentation
Either manual segmentation, or segmentation by any DL model (in this case [HPACellSegmentator](https://github.com/CellProfiling/HPA-Cell-Segmentation) for inference; training code is currently in private repo). I've also provided here example of training and segmenting dataset by the popular [cellpose v2.0](https://github.com/MouseLand/cellpose) (credits to their starter notebook, I only wrapped them in a more comprehensible/concise manner). The training set for this part is only 9 images/FOVs.

### s1 - process image masks of multiple cells to single cell masks of cell and nucleus, and into .npy

Folder: [segmentation](/segmentation) 

Removing cells where nucleus touching the borders. Cells where cell segmentation touching the bordered are still kept (maybe do a percentage rules to remove them in the future).
```sh
python s1_get_single_cell_shapes.py
```

### s2 - get FFT coeficients for individua cell and nucleus shapes

Folder: [coefficients](/coefficients) 

- Alignment and center: major axis, nuclei-cell centroid vector, major axis + nuclei centroid (mass) alignment
- Calculate FFT of x,y of the nucleus and cell segmentation (equally spaced sample along the shapes): fast fourier coefficients, elliptical fourier discriptors, wavelet
- Save result of multiprocessing pool

```sh
python s2_calculate_fft.py
```

### s3 - Calculate shape modes & map of single-organelle protein

Folder: [shapemodes](shapemodes) 

Fit and transform PCA, calculate shapemodes (n_PCs with xx% variance) based on coefficients produced from s2.
```sh
python s3_calculate_shapemodes.py
```

### s4 - Protein parameterization: Intepolate concentric rings in green channels and shape modes

Folder: [warp](warp/) 

Protein parameterization based on concentric rings from nucleus centroid - nucleus membrane - cell membrane. Final shape for all proteins: (n_rings, n_points)
```sh
python s4_concentric_rings_intensity.py
```
OR
Protein morphing on to shape based on [thin-plate splines](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=24792) given landmarks: nucleus centroid, 32p in nucleus membrane, 32p cell membrane. Final shape for all proteins = shape of the average cell in that shapemode (bin).
```sh
python s4_protein_image_warp.py
```

### s5 - Organelle distribution and relation with each other

```sh
python s5_organelle_heatmappy
```

The pilot was performed for a small subset of U2OS cell line (private images, manual segmentation and annotations): cells (1776 images) as well as HPA (public images, automatic segmentation): 297108 cells (23272 images). For historical reason, some files contained fixed paths, such as [shapemode_pipeline.py](shapemode_pipeline.py).
