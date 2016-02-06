#!/bin/bash -l
run_dir=$1
caffe test -model network/deconv-net/valid_fewer_parameter.prototxt -weights network/deconv-net/snapshots/fewer_parameters_deconvnet_for_climate_iter_7500.caffemodel -iterations 1
./scripts/plot_prob_maps.py network/deconv-net/results/fewer_params_final_data_one_blob.h5 $run_dir/plots 10