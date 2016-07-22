#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p regular
#SBATCH -t 03:00:00
#SBATCH -o batch_outputs/slurm-%A.out

./hur_main.sh $@