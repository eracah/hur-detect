


# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#Thanks, Paul!^
import sys
import math
from collections import namedtuple

import numpy as np
import tensorflow as tf



if __name__ == "__main__":
    sys.path.append("../../../")

import dotpy_src.tf_extended as tfe
from dotpy_src.box_encode_decode.configs import configs



def make_anchors_for_one_fmap(img_shape, feat_shape, dtype=np.float32):
    ind = configs["feat_shapes"].index(feat_shape)
    return ssd_anchor_one_layer(img_shape,
                      feat_shape,
                      configs["anchor_sizes"][ind],
                      configs["anchor_ratios"][ind],
                      configs["anchor_steps"][ind],
                      configs["anchor_offset"],
                     dtype=np.float32)



# ======================================================================= #
def make_anchors(img_shape, dtype=np.float32):
    """Compute the default anchor boxes, given an image shape.
    """
    return ssd_anchors_all_layers(img_shape,
                                  configs["feat_shapes"],
                                  configs["anchor_sizes"],
                                  configs["anchor_ratios"],
                                  configs["anchor_steps"],
                                  configs["anchor_offset"],
                                  dtype)



def ssd_anchors_all_layers(img_shape,
                       layers_shape,
                       anchor_sizes,
                       anchor_ratios,
                       anchor_steps,
                       offset=0.5,
                       dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors



def ssd_anchor_one_layer(img_shape,
                     feat_shape,
                     sizes,
                     ratios,
                     step,
                     offset=0.5,
                     dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w









