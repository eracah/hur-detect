


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
"""TF Extended: additional bounding boxes methods.
"""
import sys
import numpy as np
import tensorflow as tf
if __name__ == "__main__":
    sys.path.append("../../")
from dotpy_src.tf_extended import tensors as tfe_tensors
from dotpy_src.tf_extended import math as tfe_math



def bboxes_sort(scores, bboxes, top_k=400, scope=None):
    """Sort bounding boxes by decreasing order and keep only the top_k.
    If inputs are dictionnaries, assume every key is a different class.
    Assume a batch-type input.

    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      top_k: Top_k boxes to keep.
    Return:
      scores, bboxes: Sorted Tensors/Dictionaries of shape Batch x Top_k x 1|4.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_sort_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_sort(scores[c], bboxes[c], top_k=top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_sort', [scores, bboxes]):
        # Sort scores...
        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)

        # Trick to be able to use tf.gather: map for each element in the first dim.
        def fn_gather(bboxes, idxes):
            bb = tf.gather(bboxes, idxes)
            return [bb]
        r = tf.map_fn(lambda x: fn_gather(x[0], x[1]),
                      [bboxes, idxes],
                      dtype=[bboxes.dtype],
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        bboxes = r[0]
        return scores, bboxes


def bboxes_nms(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Should only be used on single-entries. Use batch version otherwise.

    Args:
      scores: N Tensor containing float scores.
      bboxes: N x 4 Tensor containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      classes, scores, bboxes Tensors, sorted by score.
        Padded with zero if necessary.
    """
    with tf.name_scope(scope, 'bboxes_nms_single', [scores, bboxes]):
        # Apply NMS algorithm.
        idxes = tf.image.non_max_suppression(bboxes, scores,
                                             keep_top_k, nms_threshold)
        scores = tf.gather(scores, idxes)
        bboxes = tf.gather(bboxes, idxes)
        # Pad results.
        scores = tfe_tensors.pad_axis(scores, 0, keep_top_k, axis=0)
        bboxes = tfe_tensors.pad_axis(bboxes, 0, keep_top_k, axis=0)
        return scores, bboxes


def bboxes_nms_batch(scores, bboxes, nms_threshold=0.5, keep_top_k=200,
                     scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Use only on batched-inputs. Use zero-padding in order to batch output
    results.

    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      scores, bboxes Tensors/Dictionaries, sorted by score.
        Padded with zero if necessary.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_nms_batch_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_nms_batch(scores[c], bboxes[c],
                                        nms_threshold=nms_threshold,
                                        keep_top_k=keep_top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_nms_batch'):
        r = tf.map_fn(lambda x: bboxes_nms(x[0], x[1],
                                           nms_threshold, keep_top_k),
                      (scores, bboxes),
                      dtype=(scores.dtype, bboxes.dtype),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        scores, bboxes = r
        return scores, bboxes


# def bboxes_fast_nms(classes, scores, bboxes,
#                     nms_threshold=0.5, eta=3., num_classes=21,
#                     pad_output=True, scope=None):
#     with tf.name_scope(scope, 'bboxes_fast_nms',
#                        [classes, scores, bboxes]):

#         nms_classes = tf.zeros((0,), dtype=classes.dtype)
#         nms_scores = tf.zeros((0,), dtype=scores.dtype)
#         nms_bboxes = tf.zeros((0, 4), dtype=bboxes.dtype)


def bboxes_matching(label, scores, bboxes,
                    glabels, gbboxes, gdifficults,
                    matching_threshold=0.5, scope=None):
    """Matching a collection of detected boxes with groundtruth values.
    Does not accept batched-inputs.
    The algorithm goes as follows: for every detected box, check
    if one grountruth box is matching. If none, then considered as False Positive.
    If the grountruth box is already matched with another one, it also counts
    as a False Positive. We refer the Pascal VOC documentation for the details.

    Args:
      rclasses, rscores, rbboxes: N(x4) Tensors. Detected objects, sorted by score;
      glabels, gbboxes: Groundtruth bounding boxes. May be zero padded, hence
        zero-class objects are ignored.
      matching_threshold: Threshold for a positive match.
    Return: Tuple of:
       n_gbboxes: Scalar Tensor with number of groundtruth boxes (may difer from
         size because of zero padding).
       tp_match: (N,)-shaped boolean Tensor containing with True Positives.
       fp_match: (N,)-shaped boolean Tensor containing with False Positives.
    """
    with tf.name_scope(scope, 'bboxes_matching_single',
                       [scores, bboxes, glabels, gbboxes]):
        rsize = tf.size(scores)
        rshape = tf.shape(scores)
        rlabel = tf.cast(label, glabels.dtype)
        # Number of groundtruth boxes.
        gdifficults = tf.cast(gdifficults, tf.bool)

        n_gbboxes = tf.count_nonzero(tf.logical_and(tf.equal(glabels, label),
                                                    tf.logical_not(gdifficults)))
        
        
        # Grountruth matching arrays.
        gmatch = tf.zeros(tf.shape(glabels), dtype=tf.bool)
        grange = tf.range(tf.size(glabels), dtype=tf.int32)
        # True/False positive matching TensorArrays.
        sdtype = tf.bool
        ta_tp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)
        ta_fp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)

        # Loop over returned objects.
        def m_condition(i, ta_tp, ta_fp, gmatch):
            r = tf.less(i, rsize)
            return r

        def m_body(i, ta_tp, ta_fp, gmatch):

            # Jaccard score with groundtruth bboxes.
            rbbox = bboxes[i]
            #returns the iou of the rbbox with every ground truth box
            jaccard = bboxes_jaccard(rbbox, gbboxes)

            jaccard = jaccard * tf.squeeze(tf.cast(tf.equal(glabels, rlabel), dtype=jaccard.dtype), axis=1)
            # Best fit, checking it's above threshold.
            idxmax = tf.cast(tf.argmax(jaccard, axis=0), tf.int32)
            jcdmax = jaccard[idxmax]
            match = jcdmax > matching_threshold
            existing_match = gmatch[idxmax]
            not_difficult = tf.logical_not(gdifficults[idxmax])

            # TP: match & no previous match and FP: previous match | no match.
            # If difficult: no record, i.e FP=False and TP=False.
            tp = tf.logical_and(not_difficult,
                                tf.logical_and(match, tf.logical_not(existing_match)))
            ta_tp = ta_tp.write(i, tp)
            fp = tf.logical_and(not_difficult,
                                tf.logical_or(existing_match, tf.logical_not(match)))
            ta_fp = ta_fp.write(i, fp)
            # Update grountruth match.
            mask = tf.logical_and(tf.equal(grange, idxmax),
                                  tf.logical_and(not_difficult, match))
            
            mask = tf.expand_dims(mask,axis=1)
            gmatch = tf.logical_or(gmatch, mask)
            return [i+1, ta_tp, ta_fp, gmatch]
        

        # Main loop definition.
        i = 0
        [i, ta_tp_bool, ta_fp_bool, gmatch] =             tf.while_loop(m_condition, m_body,
                          [i, ta_tp_bool, ta_fp_bool, gmatch],
                          parallel_iterations=1,
                          back_prop=False)
        # TensorArrays to Tensors and reshape.
        tp_match = tf.reshape(ta_tp_bool.stack(), rshape)
        fp_match = tf.reshape(ta_fp_bool.stack(), rshape)

        # Some debugging information...
        # tp_match = tf.Print(tp_match,
        #                     [n_gbboxes,
        #                      tf.reduce_sum(tf.cast(tp_match, tf.int64)),
        #                      tf.reduce_sum(tf.cast(fp_match, tf.int64)),
        #                      tf.reduce_sum(tf.cast(gmatch, tf.int64))],
        #                     'Matching (NG, TP, FP, GM): ')
        return n_gbboxes, tp_match, fp_match


def bboxes_matching_batch(labels, scores, bboxes,
                          glabels, gbboxes, gdifficults,
                          matching_threshold=0.5, scope=None):
#     if isinstance(labels, int) or isinstance(labels, float):
#         print labels, scores, bboxes, glabels, gbboxes, gdifficults,
        
    """Matching a collection of detected boxes with groundtruth values.
    Batched-inputs version.

    Args:
      rclasses, rscores, rbboxes: BxN(x4) Tensors. Detected objects, sorted by score;
      glabels, gbboxes: Groundtruth bounding boxes. May be zero padded, hence
        zero-class objects are ignored.
      matching_threshold: Threshold for a positive match.
    Return: Tuple or Dictionaries with:
       n_gbboxes: Scalar Tensor with number of groundtruth boxes (may difer from
         size because of zero padding).
       tp: (B, N)-shaped boolean Tensor containing with True Positives.
       fp: (B, N)-shaped boolean Tensor containing with False Positives.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_matching_batch_dict'):
            d_n_gbboxes = {}
            d_tp = {}
            d_fp = {}
            for c in labels:
                n, tp, fp, _ = bboxes_matching_batch(c, scores[c], bboxes[c],
                                                     glabels, gbboxes, gdifficults,
                                                     matching_threshold)
                d_n_gbboxes[c] = n
                d_tp[c] = tp
                d_fp[c] = fp
            return d_n_gbboxes, d_tp, d_fp, scores

    with tf.name_scope(scope, 'bboxes_matching_batch',
                       [scores, bboxes, glabels, gbboxes]):
        r = tf.map_fn(lambda x: bboxes_matching(labels, x[0], x[1],
                                                x[2], x[3], x[4],
                                                matching_threshold),
                      (scores, bboxes, glabels, gbboxes, gdifficults),
                      dtype=(tf.int64, tf.bool, tf.bool),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=True,
                      infer_shape=True)
        return r[0], r[1], r[2], scores


# =========================================================================== #
# Standard boxes computation.
# =========================================================================== #
def bboxes_jaccard(bbox_ref, bboxes, name=None):
    """Compute jaccard score between a reference box and a collection
    of bounding boxes.

    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with Jaccard scores.
    """
    with tf.name_scope(name, 'bboxes_jaccard'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = -inter_vol             + (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])             + (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
        jaccard = tfe_math.safe_divide(inter_vol, union_vol, 'jaccard')
        return jaccard





