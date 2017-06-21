


import sys
import numpy as np
import tensorflow as tf
if __name__ == "__main__":
    sys.path.append("../../")
from make_anchors_orig import make_anchors
from dotpy_src.configs import configs 
from dotpy_src.box_processing.tf_box_util import ious_with_anchors, convert_to_xyminmax, make_box_coords_relative,encode_to_scaled_offsets, convert_to_yxhw,make_actual_gt_box_mask, zero_out_negatives, extract_box_coordinates
import numpy as np
from make_anchors_orig import make_anchors



def encode(bboxes, labels):

    # do we zero out negatives here?
    labels = tf.cast(zero_out_negatives(labels), dtype=tf.int32)
    bboxes = zero_out_negatives(bboxes)
    bboxes = make_box_coords_relative(bboxes)
    ymin, ymax, xmin,xmax = extract_box_coordinates(bboxes)
    
    actual_gt_box_mask = make_actual_gt_box_mask(ymin, ymax, xmin, xmax)
    all_anchors = make_anchors()
    
    #get the shape of each anchor array -> same as fmap shapes
    fmap_shapes = [anchor[0].shape[:2] for anchor in all_anchors]
    
    #dictionary mapping fmap shape to anchors for that fmap shape
    anchors_map = dict(zip(fmap_shapes,all_anchors))
    
    encoded_boxes_dict = {}
    encoded_labels_dict = {}
    for fmap_shape, anchors in anchors_map.iteritems():
        encoded_boxes_dict[fmap_shape] = _encode_boxes(bboxes, anchors)
        encoded_labels_dict[fmap_shape] = _encode_labels(labels, anchors)
        
    return encoded_boxes_dict, encoded_labels_dict
        
        
        
        
    
    
    



def _encode_labels(labels, anchors):
    ay, ax, ah, aw = anchors
    encoded_labels = labels * tf.cast(tf.expand_dims(tf.expand_dims(tf.ones_like(ay*ah),axis=-1),axis=-1), dtype=tf.int32)
    encoded_labels = tf.transpose(encoded_labels,perm=[3,0,1,2,4])
    # eoncoded_labels is (batch_size, y, x, num_anchors, max_boxes(15))
    return encoded_labels



def expand_anchors(anchors):
    ay, ax, ah, aw = anchors
    ay, ax = [np.expand_dims(tens, axis=-1) for tens in [ay,ax]]
    return ay,ax,ah,aw
    



def _encode_boxes(bboxes, anchors):
    ymin, ymax, xmin,xmax = extract_box_coordinates(bboxes)
    cy,cx,h,w = convert_to_yxhw(ymin, ymax,xmin, xmax)
    ay, ax, ah, aw = expand_anchors(anchors)
    enc_y, enc_x, enc_h, enc_w = encode_to_scaled_offsets(cy,cx,h,w, ay, ax, ah, aw)
    
    encoded_boxes = tf.stack([enc_y, enc_x, enc_h, enc_w],axis=-1)
    encoded_boxes = tf.transpose(encoded_boxes,perm=(2,0,1,4,5,3))
    # encoded_boxes is (batch_size, y, x, num_anchors, num_coordinates(4), max_boxes(15))
    return encoded_boxes
    
    



if __name__ == "__main__":
    import h5py
    bboxes = h5py.File(configs["tr_data_file"])["boxes"][7:12]
    bbox_coords = bboxes[:,:,:4]
    labels = bboxes[:,:,4]
    ymin = encode(bbox_coords, labels)
    
    with tf.Session() as sess:
        print sess.run(ymin)
        assert False
        loc_tens = sess.run(btens)
        lab_tens = sess.run(ltens)
        for k, v in loc_tens.iteritems():
            print k
            print v









