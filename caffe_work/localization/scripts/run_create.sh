#!/bin/bash -l
max_h5_size=$1
use_neg=$2

#directory where this script stored
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

$DIR/create_climate_dataset_for_caffe.py \
 /global/project/projectdirs/nervana/yunjie/dataset/localization/larger_hurricanes_loc.h5 \
 /global/project/projectdirs/mantissa/climate/caffe_data \
~/projects/mantissa-climate/caffe_work/localization/data \
$max_h5_size \
$use_neg 

#ln -s /global/project/projectdirs/mantissa/climate/caffe_data  ~/projects/mantissa-climate/data/raw_data/caffe_data

#ln -s /global/project/projectdirs/mantissa/climate/caffe_data ~/projects/mantissa-climate/caffe_work/localization/data
