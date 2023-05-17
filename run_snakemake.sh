#!/bin/bash
#
#SBATCH --job-name=pipe
#
#SBATCH --time=25:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
module load python/3.9.0
pip install snakemake
snakemake --cluster "sbatch -t 25:00:00 -p normal -N 1" -j 1
