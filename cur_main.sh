#!/bin/bash -l

gpu=$1
shift

./run_main.sh $gpu --data_dir ~/data/climate/input --metadata_dir ~/data/climate/labels  --num_tr_days 365  $@
