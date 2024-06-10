#!/bin/bash
#
#SBATCH --job-name=imagewarp
#
#SBATCH --time=10:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G

module load python/3.9.0
module load py-pandas/2.0.1_py39
module load py-scikit-learn/1.0.2_py39
module load viz
module load py-scikit-image/0.19.3_py39
pip install more-itertools
module load py-scipy/1.6.3_py39
pip install imageio
srun -N 1 -n1 python3 avg_organelle.py --org Microtubules  --pc PC1 &
srun -N 1 -n1 python3 avg_organelle.py --org Nucleoplasm --pc PC1 &
srun -N 1 -n1 python3 avg_organelle.py --org Cytosol  --pc PC1 &
srun -N 1 -n1 python3 avg_organelle.py --org Vesicles  --pc PC1 &
srun -N 1 -n1 python3 avg_organelle.py --org EndoplasmicR  --pc PC1 &
srun -N 1 -n1 python3 avg_organelle.py --org NuclearM  --pc PC2 &
srun -N 1 -n1 python3 avg_organelle.py --org Nucleoli  --pc PC5 &
srun -N 1 -n1 python3 avg_organelle.py --org PlasmaM  --pc PC1 &
srun -N 1 -n1 python3 avg_organelle.py --org Mitochondria  --pc PC1 &
srun -N 1 -n1 python3 avg_organelle.py --org NuclearS  --pc PC1 &
