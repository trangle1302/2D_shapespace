#!/bin/bash
#
#SBATCH --job-name=hek
#
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G

module load python/3.9.0
module load py-pandas/1.3.1_py39
module load py-scipy/1.6.3_py39
module load py-scikit-image/0.19.3_py39
module load py-scikit-learn/1.0.2_py39
module load opencv/4.5.2
pip install imageio
pip install tqdm
time python3 s4_tsp.py --cell_line HEK_293
#python3 s4_tsp.py --cell_line A-431
#python3 s4_tsp.py --cell_line U-251_MG #A-431 #RT4
#python3 s4_tsp.py --cell_line BJ
#srun -N 1 -n1 python3 avg_organelle.py --org GolgiA  --pc PC1
#srun -N 1 -n1 python3 avg_organelle.py --org Microtubules  --pc PC1
#srun -N 1 -n1 python3 avg_protein.py --merged_bins 2  --pc PC6
#srun -N 1 -n1 python3 avg_protein.py --merged_bins 5  --pc PC6
#srun -N 1 -n1 python3 avg_protein.py --merged_bins 5  --pc PC5
