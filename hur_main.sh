#!/bin/bash -l

[ -a batch_outputs ] || mkdir batch_outputs
module load python
module load deeplearning
python hur_main.py $@