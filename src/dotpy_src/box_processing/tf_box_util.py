


import sys
if __name__ == "__main__":
    sys.path.append("../../")
import tensorflow as tf
from dotpy_src.configs import configs



def zero_out_negatives(tensor):
    return tf.where(tensor < 0, tf.zeros_like(tensor), tensor)



def extract_box_coordinates(bboxes):
    last_axis = len(bboxes.get_shape()) - 1
    ymin,ymax, xmin, xmax = tf.split(value=bboxes,num_or_size_splits=4,axis=last_axis)
    ymin,ymax, xmin, xmax = [tf.squeeze(tens) for tens in [ymin,ymax, xmin, xmax] ]
    return ymin,ymax,xmin,xmax



def make_box_coords_relative(bboxes, im_shape=None):
    #bboxes -> N,15,4 or something of that ilk 3D array (num_examples, max_boxes,number of coordinates)
    bboxes = tf.cast(x=bboxes, dtype=tf.float32)
    if not im_shape:
        im_shape = configs["input_shape"][1:]
    
    
    ymin, ymax, xmin, xmax = extract_box_coordinates(bboxes)

    ydim, xdim = im_shape
    
    ymin = ymin / float(ydim)
    ymax = ymax / float(ydim)
    xmin = xmin / float(xdim)
    xmax = xmax / float(xdim)
    
    bboxes = tf.stack([ymin, ymax,xmin,xmax], axis=-1)
    return bboxes



def convert_to_yxhw(ymin, ymax,xmin, xmax):
    # Transform to center / size.
    cy = (ymax + ymin) / 2.
    cx = (xmax + xmin) / 2.
    h = ymax - ymin
    w = xmax - xmin
    return cy,cx,h,w



def convert_to_xyminmax(y, x, h, w):
    ymin = y - h / 2.
    xmin = x - w / 2.
    ymax = y + h / 2.
    xmax = x + w / 2.
    return ymin,ymax,xmin,xmax



def encode_to_scaled_offsets(cy,cx,h,w, yref, xref, href, wref):

    cy,cx,h,w, yref, xref, href, wref = [tf.cast(arr,tf.float32) for arr in [cy,cx,h,w, yref, xref, href, wref]]
    # Encode features.
    
    cy = tf.expand_dims((cy - yref), axis=-1) / href
    cx = tf.expand_dims((cx - xref), axis=-1) / wref
    
    # some infs will be made here b/c some h's and w's are 0 and log(0) is -inf
    # we can't just mask them out b/c 0 * -inf is nan, so we set them to 0. well instead of multiplying
    # by mask, we use where or boolean mask maybe?
    h = tf.log(tf.expand_dims(h, axis=-1) / href)
    w = tf.log(tf.expand_dims(w, axis=-1) / wref)
    
    fmap_y, fmap_x = cy.shape[:2]
    empty_fmap = tf.ones(shape=(fmap_y,fmap_x,1,1,1), dtype=tf.float32)
    h = h* empty_fmap
    w = w* empty_fmap
    return cy,cx,h,w
    



def make_actual_gt_box_mask(ymin,ymax,xmin,xmax):
    both_ys_nonzero = tf.logical_and(tf.greater(ymin,0.),tf.greater(ymax,0.))
    both_xs_nonzero = tf.logical_and(tf.greater(xmin,0.),tf.greater(xmax,0.))
    actual_gt_box_mask = tf.logical_and(both_xs_nonzero, both_ys_nonzero)
    return actual_gt_box_mask



def ious_with_anchors(bbox, anchors):
    """Compute jaccard score between a box and the anchors.
        bbox: one box of [bymin, bymax, bxmin, bxmax]
        anchors: a list of arrays [aymin,aymax,axmin,axmax]
    """
    bymin, bymax, bxmin, bxmax = bbox
    aymin, aymax, axmin, axmax = anchors
    
    int_ymin = tf.maximum(aymin, bymin)
    int_xmin = tf.maximum(axmin, bxmin)
    int_ymax = tf.minimum(aymax, bymax)
    int_xmax = tf.minimum(axmax, bxmax)
    h = tf.maximum(int_ymax - int_ymin, 0.)
    w = tf.maximum(int_xmax - int_xmin, 0.)
    
    anchors_areas = (aymax - aymin) * (axmax - axmin)
    bbox_area = (bymax - bymin) * (bxmax - bxmin)

    
    intersections = h * w
    union = anchors_areas + bbox_area - intersections
    iou = tf.div(intersections, union)
    return iou



def convert_tf_shape_to_int_tuple(tf_shape):
    return tuple([dim.value for dim in tf_shape])



def zero_out_negative_rows(tensor):
    """takes Nx15x5 tensor"""
    tmask = tf.cast(tensor >= 0, dtype=tensor.dtype)
    tensor = tf.multiply(tmask, tensor)
    return tensor
    
    

def split_boxes_labels(label_tensor, axis=1):
    bboxes, classes = tf.split(label_tensor, axis=axis,num_or_size_splits=[4,1])
    return bboxes, classes

def mask_out_negative_rows(tensor):
    tmask = tf.cast(tensor >= 0, dtype=tensor.dtype)
    tensor = tf.boolean_mask(mask=tmask,tensor=tensor)
    return tensor

