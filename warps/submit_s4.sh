#!/bin/bash
#
#SBATCH --job-name=imagewarp
#
#SBATCH --time=5:00:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G

module load python/3.9.0
module load py-pandas/1.3.1_py39
module load py-scipy/1.6.3_py39
module load py-numpy
module load py-scikit-image/0.19.3_py39
module load opencv/4.5.2
pip install imageio
pip install tqdm
srun -N 1 -n1 python3 avg_organelle.py --org Nucleoplasm  --pc PC3 &
srun -N 1 -n1 python3 avg_organelle.py --org Nucleoplasm  --pc PC4 &
srun -N 1 -n1 python3 avg_organelle.py --org Nucleoplasm  --pc PC5 &
srun -N 1 -n1 python3 avg_organelle.py --org Nucleoplasm  --pc PC6 &
srun -N 1 -n1 python3 avg_organelle.py --org VesiclesPCP  --pc PC4 &
srun -N 1 -n1 python3 avg_organelle.py --org Nucleoplasm  --pc PC1 &
srun -N 1 -n1 python3 avg_organelle.py --org Nucleoplasm  --pc PC2 &
srun -N 1 -n1 python3 avg_organelle.py --org VesiclesPCP  --pc PC5 &
srun -N 1 -n1 python3 avg_organelle.py --org VesiclesPCP  --pc PC1 &
srun -N 1 -n1 python3 avg_organelle.py --org VesiclesPCP  --pc PC2 &
srun -N 1 -n1 python3 avg_organelle.py --org VesiclesPCP  --pc PC3 &
srun -N 1 -n1 python3 avg_organelle.py --org VesiclesPCP  --pc PC4 &
