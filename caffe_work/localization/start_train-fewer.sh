#!/bin/bash -l
dir=$1
run_dir=`./scripts/create_run_dir.py $dir`
caffe train -solver network/deconv-net/solver_fewer.prototxt > $run_dir/train_fewer.txt
#./scripts/plot_learning_curves.py $run_dir/train_fewer.txt $run_dir/plots


