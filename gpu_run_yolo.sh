#!/bin/bash

export THEANO_FLAGS=mode=FAST_RUN,device=gpu"$1",lib.cnmem=0.95,allow_gc=True,floatX=float32,nvcc.fastmath=True,profile=False,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once
shift
python -u hur_main.py $@
