


import tensorflow as tf
import sys
if __name__ == "__main__":
    sys.path.append("../../")
from dotpy_src.configs import configs
from dotpy_src.box_processing.tf_box_util import make_actual_gt_box_mask,ious_with_anchors, convert_to_xyminmax, encode_to_scaled_offsets,convert_to_yxhw, make_box_coords_relative, extract_box_coordinates,zero_out_negatives
import numpy as np
from make_anchors_orig import make_anchors



def match_boxes(bboxes):
    """bboxes: numpy array B x max_boxes x 4 box coordinates for each batch, where B is number of Batches
                * optionally may have -1's where there is no box (unnormalized by im size)"""
    # zero out any negative ones
    bboxes = zero_out_negatives(bboxes)
    bboxes = make_box_coords_relative(bboxes)
    ymin, ymax, xmin,xmax = extract_box_coordinates(bboxes)
    actual_gt_box_mask = make_actual_gt_box_mask(ymin, ymax, xmin, xmax)
    #get anchors
    all_anchors = make_anchors()
    #get the shape of each anchor array -> same as fmap shapes
    fmap_shapes = [anchor[0].shape[:2] for anchor in all_anchors]
    #dictionary mapping fmap shape to anchors for that fmap shape
    anchors_map = dict(zip(fmap_shapes,all_anchors))
    mask_dict={}
    for fmap_shape, anchors in anchors_map.iteritems():
        x_mask, tp_mask, num_matches = _match_boxes(bboxes,anchors, actual_gt_box_mask)
        mask_dict[fmap_shape] = (x_mask, tp_mask, num_matches)
    return mask_dict,actual_gt_box_mask

def _match_boxes(bboxes, anchors,actual_gt_box_mask, matching_threshold=0.5):
    """bboxes: numpy array B x max_boxes x 4 box coordinates for each batch, where B is number of Batches
                   * 0's where no box, normalized 
       anchors_for_layer: list of 4 arrays (y,x,h,w)
           * y and x is fm x fn x 1
           * h and w are (M,) where M is number of anchor box shapes"""
    aymin, aymax, axmin, axmax = preprocess_anchors(anchors)
    ymin, ymax, xmin,xmax = extract_box_coordinates(bboxes)
    ious = ious_with_anchors(anchors=[aymin, aymax, axmin, axmax], bbox=[ymin,ymax,xmin,xmax])
    with tf.name_scope("ious"):
        tf.summary.histogram("ious", tf.reshape(ious,[-1]))
    x_mask, tp_mask = make_x_mask(ious, actual_gt_box_mask, matching_threshold)
    float_x_mask = tf.cast(x_mask,dtype=tf.float32)        
    num_matches = tf.reduce_sum(float_x_mask)
    return x_mask, tp_mask, num_matches

def preprocess_anchors(anchors):
    # anchor processing
    ay, ax, ah, aw = anchors
    aymin, aymax, axmin, axmax= convert_to_xyminmax(ay, ax, ah, aw)
    #pad with two dims at end of 1
    aymin, aymax, axmin, axmax = [np.expand_dims(np.expand_dims(tens,axis=-1), axis=-1)                                  for tens in [aymin, aymax, axmin, axmax]]
    return aymin, aymax, axmin, axmax

def make_x_mask(ious, actual_gt_box_mask, matching_threshold):
    

    max_iou_for_each_box = tf.reduce_max(ious, axis=[0,1,2])

    best_box_mask = tf.greater_equal(x=ious, y=max_iou_for_each_box)

    thresh_mask = tf.greater_equal(x=ious, y=matching_threshold)

    tp_mask = tf.logical_or(thresh_mask, best_box_mask)

    x_mask = tf.logical_and(actual_gt_box_mask, tp_mask)

    x_mask = tf.transpose(x_mask, perm=[3,0,1,2,4])

    #X_mask is (batch_size, y, x, num_anchors, max_boxes(15))
    return x_mask, tp_mask



def save_make_mask_im(mask, name, num_splits, split_axis):
    mask_ims = tf.split(mask,num_or_size_splits=num_splits, axis=split_axis)
    for i,mask_im in enumerate(mask_ims):
        tf.summary.image(name=name + "_"+ str(i),tensor=mask_im)



if __name__ == "__main__":
    import h5py
    bbox = h5py.File(configs["tr_data_file"])["boxes"][7:12,:,:4]
    x,am= match_boxes(bbox)
    
    with tf.Session() as sess:
        print sess.run(x[(96,144)]).shape

