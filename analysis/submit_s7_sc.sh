#!/bin/bash
#
#SBATCH --job-name=pc_var_sc
#
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=30G

module load python/3.6.1
module load py-pandas/1.0.3_py36
module load py-scikit-learn/0.24.2_py36
module load py-numpy
module load viz 
module load py-scikit-image/0.17.2_py36
pip install seaborn
srun -n 1 python3 s7_protein_covariance_sc.py --PC 1 
srun -n 1 python3 s7_protein_covariance_sc.py --PC 3 
srun -n 1 python3 s7_protein_covariance_sc.py --PC 4

srun -n 1 python3 s7_protein_covariance_sc.py --PC 2
srun -n 1 python3 s7_protein_covariance_sc.py --PC 5
srun -n 1 python3 s7_protein_covariance_sc.py --PC 6