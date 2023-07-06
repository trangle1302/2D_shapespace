#!/bin/bash
#
#SBATCH --job-name=organelle
#
#SBATCH --time=5:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G

module load python/3.6.1
module load py-pandas/1.0.3_py36
module load py-scikit-learn/0.24.2_py36
module load viz 
module load py-scikit-image/0.17.2_py36
pip install imageio
pip install seaborn
srun -n 1 python3 organelle_heatmap.py --cell_line A-431
srun -n 1 python3 organelle_heatmap.py --cell_line HEK_293
srun -n 1 python3 organelle_heatmap.py --cell_line MCF7
srun -n 1 python3 organelle_heatmap.py --cell_line U-251_MG
srun -n 1 python3 organelle_heatmap.py --cell_line HEL
srun -n 1 python3 organelle_heatmap.py --cell_line K-562
srun -n 1 python3 organelle_heatmap.py --cell_line RT4
#srun -n 1 python3 organelle_heatmap.py --cell_line BJ
