#!/bin/bash
#
#SBATCH --job-name=coef
#
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G

module load python/3.9.0
module load py-pandas/2.0.1_py39
module load py-scikit-learn/1.0.2_py39
module load viz 
module load py-scikit-image/0.19.3_py39
pip install tqdm
pip install more-itertools
python3 s2_calculate_fft.py
