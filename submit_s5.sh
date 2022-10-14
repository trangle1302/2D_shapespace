#!/bin/bash
#
#SBATCH --job-name=organelle
#
#SBATCH --time=5:00:00
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G

module load python/3.6.1
module load py-pandas/1.0.3_py36
module load py-scikit-learn/0.24.2_py36
module load py-numpy
module load viz 
module load py-scikit-image/0.17.2_py36
pip install tqdm
pip install more-itertools
srun -n 1 python3 s5_organelle_heatmap.py --org ActinF
srun -n 1 python3 s5_organelle_heatmap.py --org Aggresome
srun -n 1 python3 s5_organelle_heatmap.py --org IntermediateF
srun -n 1 python3 s5_organelle_heatmap.py --org Microtubules
srun -n 1 python3 s5_organelle_heatmap.py --org VesiclesPCP
srun -n 1 python3 s5_organelle_heatmap.py --org EndoplasmicR
srun -n 1 python3 s5_organelle_heatmap.py --org GolgiA
srun -n 1 python3 s5_organelle_heatmap.py --org Mitochondria
srun -n 1 python3 s5_organelle_heatmap.py --org NuclearM