#!/bin/bash
#
#SBATCH --job-name=corr
#
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20G

module load python/3.6.1
module load py-pandas/1.0.3_py36
module load py-scikit-learn/0.24.2_py36
module load py-numpy
module load viz 
module load py-scikit-image/0.17.2_py36
pip install seaborn
pip install deepgraph

srun -n 1 python3 dg_fast_parallel_correlation.py