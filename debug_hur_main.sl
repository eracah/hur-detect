#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p debug
#SBATCH -t 30
#SBATCH -o batch_outputs/slurm-%A.out

[ -a batch_outputs ] || mkdir batch_outputs
epochs=${1-30}
lr=${2-0.0001}
module load python
source activate deeplearning
python hur_main.py -e $epochs -l $lr