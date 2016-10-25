#!/bin/bash
THEANO_FLAGS=mode=NanGuardMode,NanGuardMode.nan_is_error=True,NanGuardMode.inf_is_error=True,NanGuardMode.big_is_error=True,device=gpu1,lib.cnmem=0.95,allow_gc=True,floatX=float32,profile=False \
  python -u hur_main.py $@
