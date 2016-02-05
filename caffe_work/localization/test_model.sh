#!/bin/bash -l
caffe test -model network/deconv-net/valid_fewer_parameter.prototxt -weights network/deconv-net/snapshots/fewer_parameters_deconvnet_for_climate_iter_7500.caffemodel -iterations 1
