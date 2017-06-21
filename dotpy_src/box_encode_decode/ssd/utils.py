


import tensorflow as tf



def get_boxes_labels_zero_out_holes(label_tensor):
    """takes Nx15x5 tensor"""
    bboxes, labels = split_boxes_labels(label_tensor,axis=2)
    bmask = tf.cast(bboxes > -1, dtype=bboxes.dtype)
    bboxes= tf.multiply(bmask,bboxes)
    
    #zero out negative ones
    lmask = tf.cast(labels > -1, dtype=labels.dtype)
    
    #add one to still keep 0 class objects
    labels = tf.multiply(lmask,labels + 1)
    labels=tf.cast(labels,dtype=tf.int64)
    
    return labels, bboxes
    

def split_boxes_labels(label_tensor, axis=1):
    bboxes, classes = tf.split(label_tensor, axis=axis,num_or_size_splits=[4,1])
    return bboxes, classes

def get_boxes_mask_out_holes(label_tensor):
    """takes:
         * label_tensor (a 15x5 tensor) -> box tensor for one example
       returns:
         * classes (n,) where n is number of valid boxes
         * boxes (n,4) where n is number of valid boxes and the 4 are ymin,xmin,ymax,xmax
        
    """
    bboxes, labels = split_boxes_labels(label_tensor, axis=1)
    bmask = bboxes[:,0] > -1
    bboxes= tf.boolean_mask(mask=bmask,tensor=bboxes)
    
    #zero out negative ones
    lmask = labels > -1
    labels = tf.boolean_mask(mask=lmask,tensor=labels)
    labels=tf.cast(labels,dtype=tf.int64)
    
    return labels, bboxes



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

