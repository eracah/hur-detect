#!/bin/bash -l

[ -a batch_outputs ] || mkdir batch_outputs
if [ ! -z "$INTEL_THEANO" ]
then
 module load inteltheano
else
 module load deeplearning
fi
python hur_main.py $@
