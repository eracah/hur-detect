


import tensorflow as tf
import sys

if __name__ == "__main__":
    sys.path.append("../../")
from dotpy_src.box_processing.tf_box_util import convert_tf_shape_to_int_tuple
from dotpy_src.configs import configs



def split_box_class(label_tensor):
    bboxes, labels = tf.split(label_tensor, num_or_size_splits=[4,1], axis=2)
    labels = tf.squeeze(labels,axis=2)
    return bboxes, labels



def unpack_net_output(y_preds):
    locs,logs,preds = {}, {}, {}
    for y_pred in y_preds:
        feat_shape = tuple([d.value for d in list(y_pred.get_shape()[1:3])])
        loc,log,pred = _unpack_net_output(y_pred, feat_shape)
        locs[feat_shape] = loc
        logs[feat_shape] = log
        preds[feat_shape] = pred
    
    return locs,logs,preds
        

def _unpack_net_output(pred_tensor, feat_shape):
    """split by grabbing the first loc_channels channels (as localization output stacked in top of logit one) """
    
    num_classes = configs["num_classes"]
    num_coords_per_box = 4
    num_boxes, loc_channels, cls_channels = get_channel_numbers(feat_shape, num_classes, num_coords_per_box)
    
    cutoff_axis=3
    localizations, logits = tf.split(pred_tensor, 
                                     axis=cutoff_axis,
                                     num_or_size_splits=[loc_channels, cls_channels])

    localizations = reshape_localizations(localizations, num_boxes, num_coords_per_box,cutoff_axis)
    

    logits = reshape_logits(logits, num_boxes, num_classes,cutoff_axis)
    
    predictions = tf.contrib.slim.softmax(logits)
    
    
                               
    return localizations, logits, predictions



def reshape_logits(logits, num_boxes, num_classes, cutoff_axis):
    """ logits (num_ex,ydim,xdim,cls_channels) we reshape each to be (num_ex, ydim,xdim,num_boxes, num_classes) """
    logits_shape = convert_tf_shape_to_int_tuple(logits.get_shape())
    new_logits_shape = tuple(list(logits_shape[0:cutoff_axis]) + [num_boxes, num_classes + 1])
    logits = tf.reshape(logits, shape=new_logits_shape)
    
    return logits



def reshape_localizations(localizations, num_boxes, num_coords_per_box, cutoff_axis):
    """ localizations (num_ex, ydim,xdim,loc_channels)  we reshape each to be: (num_ex, ydim,xdim,num_boxes, num_coords_per_box)"""
    
    localizations_shape = convert_tf_shape_to_int_tuple(localizations.get_shape())
    
    new_localization_shape = tuple(list(localizations_shape[0:cutoff_axis])
                                   + [num_boxes, num_coords_per_box])
    
    localizations = tf.reshape(localizations,
                               shape=new_localization_shape)
    
    return localizations
    
    



def get_channel_numbers(feat_shape, num_classes, num_coords_per_box):
    ind = configs["feat_shapes"].index(feat_shape)
    sizes, ratios = configs["anchor_sizes"][ind], configs["anchor_ratios"][ind]
    num_boxes = len(sizes)+ len(ratios)

    loc_channels = num_boxes * num_coords_per_box
    cls_channels = (num_classes + 1) * num_boxes
    
    return num_boxes, loc_channels, cls_channels



if __name__=="__main__":
    feat_shape = (96,144)

    num_classes = configs["num_classes"]
    ind = configs["feat_shapes"].index(feat_shape)
    sizes, ratios = configs["anchor_sizes"][ind], configs["anchor_ratios"][ind]
    num_boxes = len(sizes)+ len(ratios)
    num_coords_per_box = 4
    loc_channels = num_boxes * num_coords_per_box
    cls_channels = num_classes * num_boxes


    pred_tensor = tf.ones( (2,feat_shape[0],feat_shape[1],loc_channels + cls_channels))

    pred_tensor.get_shape()

    loc, log, pred = _unpack_net_output(pred_tensor,feat_shape)

    #assert convert_tf_shape_to_int_tuple(loc.get_shape()) == (1,96,144,4,4)
    #assert convert_tf_shape_to_int_tuple(log.get_shape()) == (1,96,144,4,4)

