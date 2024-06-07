#!/bin/bash
#
#SBATCH --job-name=pca
#
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=80G

module load python/3.9.0
module load py-pandas/2.0.1_py39
module load py-scikit-learn/1.0.2_py39
module load py-scikit-image/0.20.0_py39
#pip install tqdm
#pip install more-itertools
#python3 s3_calculate_shapemodes_ICA.py
#python3 s3_calculate_shapemodes_all.py
python3 s3_calculate_shapemodes_fucci.py
#python3 s3_calculate_shapemodes_fucci_ICA.py
#python3 s3_calculate_shapemodes.py
#pip install POT
#pip install pymanopt autograd
#python3 LDA.py
