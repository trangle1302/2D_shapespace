#!/bin/bash
#
#SBATCH --job-name=pca
#
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=30G

module load python/3.6.1
module load py-pandas/1.0.3_py36
module load py-scikit-learn/0.24.2_py36
module load py-numpy
module load viz 
module load py-scikit-image/0.17.2_py36
pip install tqdm
pip install more-itertools
python3 s3_calculate_shapemodes.py
