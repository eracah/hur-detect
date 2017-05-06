


# Copyright 2017 Paul Balanca. All Rights Reserved.
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
"""TF Extended: additional metrics.
"""

import sys
import tensorflow as tf
import numpy as np

from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
if __name__ == "__main__":
    sys.path.append("../../")
from dotpy_src.tf_extended import math as tfe_math




def _safe_div(numerator, denominator, name):
    """Divides two values, returning 0 if the denominator is <= 0.
    Args:
      numerator: A real `Tensor`.
      denominator: A real `Tensor`, with dtype matching `numerator`.
      name: Name for the returned op.
    Returns:
      0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return tf.where(
        math_ops.greater(denominator, 0),
        math_ops.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)




# =========================================================================== #
# TF Extended metrics: TP and FP arrays.
# =========================================================================== #
def precision_recall(num_gbboxes, num_detections, tp, fp, scores,
                     dtype=tf.float64, scope=None):
    """Compute precision and recall from scores, true positives and false
    positives booleans arrays
    """
    if isinstance(scores, dict):
        d_precision = {}
        d_recall = {}
        for c in num_gbboxes.keys():
            scope = 'precision_recall_%s' % c
            p, r = precision_recall(num_gbboxes[c], num_detections[c],
                                    tp[c], fp[c], scores[c],
                                    dtype, scope)
            d_precision[c] = p
            d_recall[c] = r
        return d_precision, d_recall

    # Sort by score.
    with tf.name_scope(scope, 'precision_recall',
                       [num_gbboxes, num_detections, tp, fp, scores]):
        # Sort detections by score.
        scores, idxes = tf.nn.top_k(scores, k=num_detections, sorted=True)
        tp = tf.gather(tp, idxes)
        fp = tf.gather(fp, idxes)
        # Computer recall and precision.
        tp = tf.cumsum(tf.cast(tp, dtype), axis=0)
        fp = tf.cumsum(tf.cast(fp, dtype), axis=0)
        recall = _safe_div(tp, tf.cast(num_gbboxes, dtype), 'recall')
        precision = _safe_div(tp, tp + fp, 'precision')
        return tf.tuple([precision, recall])



# =========================================================================== #
# Average precision computations.
# =========================================================================== #
def average_precision_voc12(precision, recall, name=None):
    """Compute (interpolated) average precision from precision and recall Tensors.

    The implementation follows Pascal 2012 and ILSVRC guidelines.
    See also: https://sanchom.wordpress.com/tag/average-precision/
    """
    with tf.name_scope(name, 'average_precision_voc12', [precision, recall]):
        # Convert to float64 to decrease error on Riemann sums.
        precision = tf.cast(precision, dtype=tf.float64)
        recall = tf.cast(recall, dtype=tf.float64)

        # Add bounds values to precision and recall.
        precision = tf.concat([[0.], precision, [0.]], axis=0)
        recall = tf.concat([[0.], recall, [1.]], axis=0)
        # Ensures precision is increasing in reverse order.
        precision = tfe_math.cummax(precision, reverse=True)

        # Riemann sums for estimating the integral.
        # mean_pre = (precision[1:] + precision[:-1]) / 2.
        mean_pre = precision[1:]
        diff_rec = recall[1:] - recall[:-1]
        ap = tf.reduce_sum(mean_pre * diff_rec)
        return ap


def average_precision_voc07(precision, recall, name=None):
    """Compute (interpolated) average precision from precision and recall Tensors.

    The implementation follows Pascal 2007 guidelines.
    See also: https://sanchom.wordpress.com/tag/average-precision/
    """
    with tf.name_scope(name, 'average_precision_voc07', [precision, recall]):
        # Convert to float64 to decrease error on cumulated sums.
        precision = tf.cast(precision, dtype=tf.float64)
        recall = tf.cast(recall, dtype=tf.float64)
        # Add zero-limit value to avoid any boundary problem...
        precision = tf.concat([precision, [0.]], axis=0)
        recall = tf.concat([recall, [np.inf]], axis=0)

        # Split the integral into 10 bins.
        l_aps = []
        for t in np.arange(0., 1.1, 0.1):
            mask = tf.greater_equal(recall, t)
            v = tf.reduce_max(tf.boolean_mask(precision, mask))
            l_aps.append(v / 11.)
        ap = tf.add_n(l_aps)
        return ap



class EpochMetrics(object):
    def __init__(self):
        self.num_gbboxes = {}
        self.tp = {}
        self.fp = {}
        self.rscores = {}
    def update_metrics(self,num_gbboxes, tp, fp, rscores):
        self.num_gbboxes = self.increment_dict(self.num_gbboxes, num_gbboxes)
        self.tp = self.extend_to_dict(self.tp, tp)
        self.fp = self.extend_to_dict(self.fp, fp)
        self.rscores = self.extend_to_dict(self.rscores, rscores)


    def extend_to_dict(self,main_dict,extending_dict):
        if len(main_dict.keys()) == 0:
            main_dict.update({k:[] for k in extending_dict.keys()})
        for k,v in extending_dict.iteritems():
            k_list = [list(arr) for arr in extending_dict[k]]
            for sublist in k_list:
                main_dict[k].extend(sublist)
        return main_dict
    def increment_dict(self,main_dict, incrementing_dict):
        if len(main_dict.keys()) == 0:
            main_dict.update(incrementing_dict)
            for k,v in main_dict.iteritems():
                # for lists of num_classes for each batch
                main_dict[k] = sum(v)
        else:
            for k,v in incrementing_dict.iteritems():
                main_dict[k] += sum(v)
        return main_dict
    def get_final_metrics(self):
        num_detections = {c:len(v) for c,v in self.rscores.iteritems()}
        return self.num_gbboxes,  num_detections, self.tp, self.fp, self.rscores

