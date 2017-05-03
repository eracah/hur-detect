


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
import dotpy_src.tf_extended as tfe
from dotpy_src.losses.util import convert_tf_shape_to_int_tuple
from dotpy_src.load_data.get_generator import get_generator
from dotpy_src.load_data.configs import configs as data_configs
from dotpy_src.models.configs import configs as model_configs
from dotpy_src.box_encode_decode.configs import configs as configs
configs.update(data_configs)
configs.update(model_configs)



def encode_one_fmap(label_tensor,feat_shape,im_shape=configs["tensor_input_shape"]):
    """encodes one feature map size for all examples"""
    anchors = make_anchors_for_one_fmap(img_shape=im_shape, feat_shape=feat_shape)
    
    num_examples_per_batch = configs["batch_size"]
    
    all_feat_labels, all_feat_localizations, all_feat_scores = [] , [], []
    
    for example_ind in range(num_examples_per_batch):
        #grab each box_coord thing
        _, cur_label_tensor, _ = tf.split(label_tensor, num_or_size_splits=(example_ind, 1, num_examples_per_batch - example_ind -1 ))
        cur_label_tensor = tf.squeeze(cur_label_tensor, axis=0)
        feat_labels, feat_localizations, feat_scores = encode_one_fmap_one_example(cur_label_tensor, anchors)

        
        all_feat_labels.append(feat_labels)

        all_feat_localizations.append(feat_localizations)
        all_feat_scores.append(feat_scores)
    
    
    feat_labels_tensor = tf.stack(tuple(all_feat_labels), axis=0)
    feat_localizations_tensor = tf.stack(tuple(all_feat_localizations), axis=0)
    feat_scores_tensor = tf.stack(tuple(all_feat_scores), axis=0)
    
        
    return feat_labels_tensor, feat_localizations_tensor, feat_scores_tensor



def encode_one_fmap_one_example(one_example_label_tensor, anchors):
    classes, bboxes = encode_prep(one_example_label_tensor)

    feat_labels, feat_localizations, feat_scores = tf_ssd_bboxes_encode_layer(classes,
                                                   bboxes,
                                                   anchors,
                                                   num_classes=configs["num_classes"],
                                                   no_annotation_label=True,
                                                   ignore_threshold=0.5,
                                                   prior_scaling=configs["prior_scaling"],
                                                   dtype=tf.float32)
    
    
    
    return feat_labels, feat_localizations, feat_scores
    



def encode_prep(label_tensor):
    """takes:
         * label_tensor (a 15x5 tensor) -> box tensor for one example
       returns:
         * classes (n,) where n is number of valid boxes
         * boxes (n,4) where n is number of valid boxes and the 4 are ymin,xmin,ymax,xmax
        
    """
    bboxes, classes = tf.split(label_tensor, axis=1,num_or_size_splits=[4,1])
    bmask = bboxes[:,0] > -1
    bboxes= tf.boolean_mask(mask=bmask,tensor=bboxes)
    
    #zero out negative ones
    lmask = classes > -1
    classes = tf.boolean_mask(mask=lmask,tensor=classes)
    classes=tf.cast(classes,dtype=tf.int64)
    
#     print bboxes.get_shape()
#     print classes.get_shape()
    return classes, bboxes



def encode(label, im_shape=(768.,1152.)):
    
    """currently we need to know how many boxes there are. TODO: fix that
    expects boxes to be normalized by size of image
    takes boxes for one example -> not sure what format yet
            and the shape of the images
            and encodes it into the proper ground truth tensor or list of tensors"""
    
    #make anchors
    anchors = make_anchors(im_shape)
    
    classes, bboxes = encode_prep(label)
    
    #encode boxes
    target_labels, target_localizations, target_scores = bboxes_encode(classes, bboxes, anchors,scope=None)
    
    return target_labels, target_localizations, target_scores#, labels, bboxes



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



def tf_ssd_bboxes_encode_layer(labels,
                               bboxes,
                               anchors_layer,
                               num_classes,
                               no_annotation_label,
                               ignore_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
    """Encode groundtruth labels and bounding boxes using SSD anchors from
    one layer.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores): Target Tensors.
    """
    
    #print labels
    #print bboxes
    # Anchors coordinates and volume.
    ymin_ind,ymax_ind, xmin_ind,  xmax_ind = range(4)
    
    yref, xref, href, wref = anchors_layer
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)
    
    shape = (yref.shape[0], yref.shape[1], href.size)
    feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """
  
        int_ymin = tf.maximum(ymin, bbox[ymin_ind])
        int_xmin = tf.maximum(xmin, bbox[xmin_ind])
        int_ymax = tf.minimum(ymax, bbox[ymax_ind])
        int_xmax = tf.minimum(xmax, bbox[xmax_ind])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol             + (bbox[ymax_ind] - bbox[ymin_ind]) * (bbox[xmax_ind] - bbox[xmin_ind])
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard

    def intersection_with_anchors(bbox):
        """Compute intersection between score a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[ymin_ind])
        int_xmin = tf.maximum(xmin, bbox[xmin_ind])
        int_ymax = tf.minimum(ymax, bbox[ymax_ind])
        int_xmax = tf.minimum(xmax, bbox[xmax_ind])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        scores = tf.div(inter_vol, vol_anchors)
        return scores

    def condition(i, feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.
        label = labels[i]
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox)
        # Mask: check threshold + scores + no annotations + num_classes.
        mask = tf.greater(jaccard, feat_scores)
        # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        mask = tf.logical_and(mask, feat_scores > -0.5)
        mask = tf.logical_and(mask, label < num_classes)
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)
        # Update values using mask.
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_ymin = fmask * bbox[ymin_ind] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[xmin_ind] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[ymax_ind] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[xmax_ind] + (1 - fmask) * feat_xmax

        # Check no annotation label: ignore these anchors...
        # interscts = intersection_with_anchors(bbox)
        # mask = tf.logical_and(interscts > ignore_threshold,
        #                       label == no_annotation_label)
        # # Replace scores by -1.
        # feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

        return [i+1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]
    # Main loop definition.
    i = 0
    [i, feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                           [i, feat_labels, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax],)
    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features.
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_labels, feat_localizations, feat_scores



if __name__ == "__main__":
    label_1 = tf.concat((tf.ones((1,4,5)), -1*tf.ones((1,11,5))),axis=1)
  
    label_2 = tf.concat((tf.ones((1,3,5)), -1*tf.ones((1,12,5))),axis=1)
    label_tensor = tf.concat((label_1, label_2),axis=0)
    feat_labels, feat_localizations, feat_scores = encode_one_fmap(label_tensor, feat_shape=(96,144))



if __name__ == "__main__":
    gen = get_generator("tr", batch_size=1)

    im, box = gen.next()

    #boxes = np.vstack(box)

    #boxes= make_box_coords_relative(boxes,im_shape=(768,1152))

    

    

    feat_labels, feat_localizations, feat_scores, labels,bboxes = encode(boxes, im_shape=(768,1152))

    with tf.Session() as sess:
        feed_dict = {bboxes:boxes[:,:4],
                    labels:boxes[:,4]}
        fl=sess.run(feat_labels,feed_dict=feed_dict)

    print fl[0][fl[0] > 0]





