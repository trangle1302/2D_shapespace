#!/bin/bash
#
#SBATCH --job-name=imagewarp
#
#SBATCH --time=10:00:00
#SBATCH --ntasks=17
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
srun -N 1 -n1 python3 avg_organelle.py --org Nucleoplasm  --pc PC6 &
srun -N 1 -n1 python3 avg_organelle.py --org Nucleoli  --pc PC6 &
srun -N 1 -n1 python3 avg_organelle.py --org NucleoliFC  --pc PC6 &
srun -N 1 -n1 python3 avg_organelle.py --org EndoplasmicR  --pc PC6 &
srun -N 1 -n1 python3 avg_organelle.py --org NuclearS  --pc PC6 &
srun -N 1 -n1 python3 avg_organelle.py --org GolgiA  --pc PC6 &
srun -N 1 -n1 python3 avg_organelle.py --org Microtubules  --pc PC6 &
srun -N 1 -n1 python3 avg_organelle.py --org Mitochondria  --pc PC6 &
srun -N 1 -n1 python3 avg_organelle.py --org VesiclesPCP  --pc PC6 &
srun -N 1 -n1 python3 avg_organelle.py --org PlasmaM  --pc PC6 &
srun -N 1 -n1 python3 avg_organelle.py --org Cytosol --pc PC6 &
srun -N 1 -n1 python3 avg_organelle.py --org NuclearS --pc PC6 &
srun -N 1 -n1 python3 avg_organelle.py --org ActinF --pc PC6  &
srun -N 1 -n1 python3 avg_organelle.py --org Centrosome --pc PC6  &
srun -N 1 -n1 python3 avg_organelle.py --org IntermediateF --pc PC6  &
srun -N 1 -n1 python3 avg_organelle.py --org NuclearM --pc PC6  &
srun -N 1 -n1 python3 avg_organelle.py --org NuclearB --pc PC6  
