#!/bin/bash -l

for size in None 1024
do
./run_create.sh $size
./run_create.sh $size w_neg
done
