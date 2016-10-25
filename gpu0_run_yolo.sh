#!/bin/bash
THEANO_FLAGS=mode=DebugMode,DebugMode.check_py=False,device=gpu0,lib.cnmem=0.95,allow_gc=True,floatX=float32,profile=False python -u hur_main.py $@
