#!/bin/bash
#
#SBATCH --job-name=pipe
#
#SBATCH --time=25:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

pip install snakemake
snakemake
#snakemake --cluster "sbatch -t 5:00:00 -p normal -N 1" -j 2
