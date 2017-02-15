#!/bin/bash

export THEANO_FLAGS=device=gpu"$1"
shift
python -u hur_main.py $@
