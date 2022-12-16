#!/bin/bash
#
#SBATCH --job-name=imagewarp
#
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G

module load python/3.9.0
module load py-pandas/1.3.1_py39
#module load py-scikit-learn/0.24.2_py36
module load py-scipy/1.6.3_py39
module load py-numpy
#module load viz 
module load py-scikit-image/0.19.3_py39
module load opencv/4.5.2
pip install imageio
pip install tqdm
#pip install more-itertools
python3 avg_organelle.py
