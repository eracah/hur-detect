#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p regular
#SBATCH -t 03:00:00

epochs=${1-30}
lr=${2-0.0001}
module load python
source activate deeplearning
python hur_main.py -e $epochs -l $lr