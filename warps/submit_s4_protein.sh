#!/bin/bash
#
#SBATCH --job-name=wp1
#
#SBATCH --time=10:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G

module load python/3.9.0
module load py-pandas/1.3.1_py39
module load py-scipy/1.6.3_py39
module load py-numpy
module load py-scikit-image/0.19.3_py39
module load opencv/4.5.2
pip install imageio
pip install tqdm

srun -N 1 -n1 python3 avg_protein.py --merged_bins 2  --pc PC1 &
srun -N 1 -n1 python3 avg_protein.py --merged_bins 3  --pc PC1
