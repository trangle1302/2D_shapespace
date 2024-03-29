#!/bin/bash
#
#SBATCH --job-name=sampling
#
#SBATCH --time=20:00:00
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G

module load python/3.9.0
module load py-pandas/1.3.1_py39
module load py-scikit-image/0.19.3_py39
module load py-scikit-learn/1.0.2_py39
pip install imageio
pip install tqdm
srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --cell_line A-431 --n_isos 10 20 &
srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --cell_line RT4 --n_isos 10 20 &
srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --cell_line K-562  --n_isos 10 20 &
srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --cell_line BJ  --n_isos 10 20 &
srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --cell_line MCF7  --n_isos 10 20 &
#srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --cell_line HEK_293  --n_isos 10 20 &
srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --cell_line HEL  --n_isos 10 20
