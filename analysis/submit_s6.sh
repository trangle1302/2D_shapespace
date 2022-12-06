#!/bin/bash
#
#SBATCH --job-name=pc_var
#
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

module load python/3.9
module load py-pandas/1.3.1_py39
module load py-scikit-learn/1.0.2_py39
module load py-numpy/1.20.3_py39
module load viz 
module load rust/1.63.0
# module load py-scikit-image/0.17.2_py36
pip install gseapy
python3 s6_find_var.py 