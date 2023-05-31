#!/bin/bash
#
#SBATCH --job-name=sampling
#
#SBATCH --time=10:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10

module load python/3.9.0
module load py-pandas/1.3.1_py39
pip install imageio
pip install tqdm
srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --org HEK_293 &
srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --org BJ &
srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --org A-431 &
srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --org HEL &
srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --org K562 &
srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --org MCF7 &
srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --org RT4 &
srun -N 1 -n1 python3 s4_concentric_rings_intensity.py --org U-251_MG 