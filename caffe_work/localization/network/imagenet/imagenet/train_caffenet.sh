#!/usr/bin/env sh

caffe train \
    --solver=localization/imagent/bvlc_reference_caffenet/solver.prototxt
