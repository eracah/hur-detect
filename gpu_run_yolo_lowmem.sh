#!/bin/bash

export THEANO_FLAGS=mode=FAST_RUN,device=gpu"$1",allow_gc=True,floatX=float32,nvcc.fastmath=True,profile=False 
shift
python -u hur_main.py $@
