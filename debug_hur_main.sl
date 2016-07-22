#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p debug
#SBATCH -t 30
#SBATCH -o batch_outputs/slurm-%A.out

./hur_main.sh $@