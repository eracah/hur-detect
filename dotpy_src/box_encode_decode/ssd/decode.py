


# Some of this code came from this license:
# Copyright 2015 Paul Balanca. All Rights Reserved.
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



import sys
import tensorflow as tf
import numpy as np
if __name__ == "__main__":
    sys.path.append("../../../")
from make_anchors_orig import make_anchors
from dotpy_src.configs import configs

from dotpy_src.postprocessing.utils import sort_some_lists_of_tensors



def decode(localizations):
    """takes net output or encoded ground truth label
    and returns list of boxes? and class?"""
    anchors = make_anchors()
    localizations = sort_some_lists_of_tensors(localizations)
    bboxes = bboxes_decode(localizations, anchors)
    return bboxes
    



def bboxes_decode(feat_localizations, anchors,
                  scope='ssd_bboxes_decode'):
    """Encode labels and bounding boxes.
    """
    return tf_ssd_bboxes_decode(
        feat_localizations, anchors,
        prior_scaling=configs["prior_scaling"],
        scope=scope)



def tf_ssd_bboxes_decode_layer(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      Tensor Nx4: ymin, xmin, ymax, xmax
    """
    yref, xref, href, wref = anchors_layer
    
    x_loc, y_loc, w_loc, h_loc = [ tf.squeeze( tens, axis=4) for tens in tf.split(feat_localizations,axis=4,num_or_size_splits=4)]
    cx = x_loc * wref * prior_scaling[0] + xref
    cy = y_loc * href * prior_scaling[1] + yref
    w = wref * tf.exp(w_loc * prior_scaling[2])
    h = href * tf.exp(h_loc * prior_scaling[3])
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes


def tf_ssd_bboxes_decode(feat_localizations,
                         anchors,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         scope='ssd_bboxes_decode'):
    """Compute the relative bounding boxes from the SSD net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx4: ymin, xmin, ymax, xmax
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_ssd_bboxes_decode_layer(feat_localizations[i],
                                           anchors_layer,
                                           prior_scaling))
        return bboxes




if __name__ == "__main__":
    from dotpy_src.postprocessing.unpack import unpack_net_output
    shapes = [(2, 6, 9, 48),
                     (2, 3, 5, 32),
                     (2, 96, 144, 32),
                     (2, 24, 36, 48),
                     (2, 12, 18, 48),
                     (2, 48, 72, 48),
                     (2, 1, 1, 32)]

    y_preds = [tf.ones((shape)) for shape in shapes]
    localizations, logits, predictions = unpack_net_output(y_preds)
    
    #print localizations
    boxes=decode(localizations)





