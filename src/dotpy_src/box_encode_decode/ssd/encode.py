


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
import numpy as np
import tensorflow as tf
if __name__ == "__main__":
    sys.path.append("../../../")
from make_anchors_orig import make_anchors,make_anchors_for_one_fmap
#import dotpy_src.tf_extended as tfe
from dotpy_src.postprocessing.utils import convert_tf_shape_to_int_tuple
from dotpy_src.load_data.get_generator import get_generator
from dotpy_src.configs import configs
from utils import get_boxes_mask_out_holes,tf_ssd_bboxes_encode_layer



def encode(label, im_shape=configs["input_shape"][-2:]):
    
    """input: label (a Nx15x5 tensor, where N is batch_size)
       output: batch_target_labels, batch_target_localizations, batch_target_scores
                 * each of these is a list of tensors, one tensor for each fmap size,
                 * 1st dim of tensor is batch size """
    
    num_examples_per_batch = configs["batch_size"]
    
    #make anchors
    anchors = make_anchors(im_shape)
    
    batch_target_labels, batch_target_localizations, batch_target_scores = [], [], []
    for example_ind in range(num_examples_per_batch):
        
        #grab each box_coord thing
        _, cur_label_tensor, _ = tf.split(label, num_or_size_splits=(example_ind, 1, num_examples_per_batch - example_ind -1 ))
        cur_label_tensor = tf.squeeze(cur_label_tensor, axis=0)
        classes, bboxes = get_boxes_mask_out_holes(cur_label_tensor)
    
        #encode boxes
        target_labels_one_example, target_localizations_one_example, target_scores_one_example = bboxes_encode(classes, bboxes, anchors,scope=None)
        
        
        batch_target_labels.append(target_labels_one_example)

        batch_target_localizations.append(target_localizations_one_example)
        batch_target_scores.append(target_scores_one_example)
        
    
    #stack each label tensor of the same fmap size for each example
    target_labels = [tf.stack(labels,axis=0) for labels in zip(*batch_target_labels)]
    target_localizations = [tf.stack(localizations,axis=0) for localizations in zip(*batch_target_localizations)] 
    target_scores = [tf.stack(scores,axis=0) for scores in zip(*batch_target_scores)] 
    
    return target_labels, target_localizations, target_scores



def bboxes_encode(labels, bboxes, anchors,
                  scope=None):
    """Encode labels and bounding boxes.
    """
    return tf_ssd_bboxes_encode(
        labels, bboxes, anchors,
        configs["num_classes"],
        no_annotation_label=True,
        ignore_threshold=0.5,
        prior_scaling=configs["prior_scaling"],
        scope=scope)




def tf_ssd_bboxes_encode(labels,
                         bboxes,
                         anchors,
                         num_classes,
                         no_annotation_label,
                         ignore_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='ssd_bboxes_encode'):
    """Encode groundtruth labels and bounding boxes using SSD net anchors.
    Encoding boxes for all feature layers.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors: List of Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores):
        Each element is a list of target Tensors.
    """
    with tf.name_scope(scope):
        target_labels = []
        target_localizations = []
        target_scores = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                t_labels, t_loc, t_scores =                     tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer,
                                               num_classes, no_annotation_label,
                                               ignore_threshold,
                                               prior_scaling, dtype)
                target_labels.append(t_labels)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)
        return target_labels, target_localizations, target_scores



if __name__ == "__main__":
    label_1 = tf.concat((tf.ones((1,4,5)), -1*tf.ones((1,11,5))),axis=1)
  
    label_2 = tf.concat((tf.ones((1,3,5)), -1*tf.ones((1,12,5))),axis=1)
    label_tensor = tf.concat((label_1, label_2),axis=0)
    #labels, bboxes = get_boxes_labels_zero_out_holes(label_tensor)
    feat_labels, feat_localizations, feat_scores = encode(label_tensor)

    with tf.Session() as sess:
        print sess.run(feat_localizations)





