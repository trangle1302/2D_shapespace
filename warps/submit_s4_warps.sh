#!/bin/bash
#
#SBATCH --job-name=HEL
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G

module load python/3.9.0
module load py-pandas/1.3.1_py39
module load py-scikit-image/0.19.3_py39
module load py-scikit-learn/1.0.2_py39
pip install imageio
pip install tqdm
python3 s4_tsp.py --cell_line U-251_MG 
python3 s4_tsp.py --cell_line MCF7 
python3 s4_tsp.py --cell_line HEL
