


import sys
import tensorflow as tf
import numpy as np
slim=tf.contrib.slim
if __name__ == "__main__":
    sys.path.append("../../../")
from dotpy_src.losses.utils import abs_smooth as smooth_L1




from dotpy_src.configs import configs
from dotpy_src.postprocessing.utils import get_int_tensor_shape, sort_some_lists_of_tensors

from dotpy_src.box_processing.match import match_boxes
from dotpy_src.box_processing.encode import encode
from dotpy_src.box_processing.unpack import unpack_net_output, split_box_class
#from utils import ssd_losses



loss_weights = None



def compute_loss(y_true, y_preds):
    '''y_true: the boxes Nx15x5 tensor
       y_preds: a list of 7?  tensors of Nxfy x fx x k where k = 4*number of anchors + number_of_anchors*num_classes,
       N is number of examples'''

    bboxes, labels = split_box_class(y_true)
    
    encoded_boxes_dict, encoded_labels_dict = encode(bboxes,labels)

    mask_dict,actual_gt_box_mask = match_boxes(bboxes)
    
    loc_dict, log_dict, pred_dict = unpack_net_output(y_preds)
    loss = tf.constant(0.)

    for fmap_size in encoded_boxes_dict.keys():
        x_mask, tp_mask, num_matches = mask_dict[fmap_size]
        with tf.name_scope("loss_for_%i_%i" %(fmap_size[0], fmap_size[1])):
            tf.summary.scalar(name="num_positives",tensor=num_matches)

    #         with tf.name_scope("x_mask"):
    #             batch_size = configs["batch_size"]
    #             max_boxes = configs["num_max_boxes"]
    #             num_anchors = 4
    #             for anch in range(num_anchors):
    #                 for gt_box in range(max_boxes):

    #                     x_im = tf.cast(tf.expand_dims(x_mask[:,:,:,anch,gt_box], axis=-1), tf.float32)
    #                     tf.summary.image("xmask_for_fmap_%i_%i_anchor_%i_gt_box_%i"%(fmap_size[0],fmap_size[1],anch,gt_box),
    #                                      x_im)

            encoded_boxes, encoded_labels, loc, log, pred = [dic[fmap_size] for dic in [encoded_boxes_dict,
                                                              encoded_labels_dict,
                                                               loc_dict, log_dict, pred_dict ]]

            fmap_loss = calc_loss_one_layer(encoded_boxes, encoded_labels, x_mask, tp_mask,
                                           num_matches,actual_gt_box_mask, loc, log, pred, fmap_size )

            loss = loss + fmap_loss
    return loss
        
        



def calc_loss_one_layer(encoded_boxes, encoded_labels, x_mask, tp_mask, num_matches,actual_gt_box_mask, loc, log, pred, fmap_size ):

    loc_loss = calc_loc_loss(x_mask, encoded_boxes, loc)
    cls_loss = calc_cls_loss(x_mask, encoded_labels,log, pred,actual_gt_box_mask, tp_mask, num_matches, fmap_size ) 
    match_coeff = tf.where(tf.equal(num_matches, 0), 0., tf.div(1., tf.cast(num_matches,dtype=tf.float32)))
    return (loc_loss + cls_loss) * match_coeff * (1./ configs["batch_size"])

def calc_loc_loss(x_mask, encoded_boxes, loc):
    num_coords_in_a_box = 4
    loc = tf.expand_dims(loc,axis=-1)
    x_mask = tf.stack(num_coords_in_a_box*[x_mask], axis=-2)
    loc_loss = smooth_L1(encoded_boxes - loc)
    
    # this gets rid of nans and infs that were encountered when having heights and widths of zero due to zeroing out -1's
    #from the labels!
    loc_loss = tf.boolean_mask(loc_loss, x_mask)
    with tf.name_scope("loc_losses"):
        loc_loss = tf.reduce_sum(loc_loss)
        tf.summary.scalar("loc_loss", loc_loss)
    return loc_loss

def calc_cls_loss(x_mask, encoded_labels,log, pred,actual_gt_box_mask, tp_mask, num_matches, fmap_size ):
    log = tf.stack(configs["num_max_boxes"]*[log], axis=-2)
    xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=log, labels=encoded_labels)
    xent_loss = tf.boolean_mask(xent_loss, x_mask)
    pos_xent = tf.reduce_sum(xent_loss)
    
    fp_mask = make_fp_mask(actual_gt_box_mask, tp_mask)

    num_negs = tf.cast(configs["negative_ratio"] * num_matches, dtype=tf.int32)
    
    default_negs = tf.constant( configs["default_negatives"],dtype=tf.int32)
    num_negs = tf.where(tf.equal(num_negs,0),default_negs, num_negs)

    tf.summary.scalar(name="num_mined_negatives",tensor=num_negs) 
    hard_neg_mask = make_hard_neg_mask(pred, fp_mask, num_negs, fmap_size)
    
    neg_encoded_labels = tf.where(hard_neg_mask, configs["num_classes"]*tf.ones_like(encoded_labels), encoded_labels)
    neg_xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=log, labels=neg_encoded_labels)
    neg_xent_loss = tf.boolean_mask(neg_xent_loss, hard_neg_mask)
    neg_xent = tf.reduce_sum(neg_xent_loss)
    with tf.name_scope("cls_losses"):
        tf.summary.scalar("pos_xent",pos_xent)
        tf.summary.scalar("neg_xent", neg_xent)
    return pos_xent + neg_xent
    
    
    



def make_hard_neg_mask(pred, fp_mask, num_negs, fmap_size):
    pred = tf.stack(15*[pred], axis=-2)
    float_fp_mask = tf.cast(fp_mask, dtype=pred.dtype)

    float_fp_mask = tf.expand_dims(float_fp_mask, axis=-1)
    fp_pred = pred * float_fp_mask
    last_axis = len(fp_pred.get_shape()) -1
    # bg class is the last element of logits
    _, bg_fp_pred = tf.split(fp_pred, num_or_size_splits=[configs["num_classes"], 1], axis=last_axis)
    flat_bg_fp_pred = tf.reshape(bg_fp_pred, [-1])
    
    num_total_negs = flat_bg_fp_pred.get_shape()[0]
    tf.summary.scalar(name="num_total_negatives", tensor=num_total_negs)

    k = tf.where(num_negs > num_total_negs, num_total_negs, num_negs)
    bg_confs, inds = tf.nn.top_k(flat_bg_fp_pred,k=k)
    min_bg_conf = bg_confs[-1]
    hard_neg_mask = tf.greater_equal(bg_fp_pred, min_bg_conf)
    hard_neg_mask = tf.squeeze(hard_neg_mask, axis=-1)

    return hard_neg_mask
    
    
    



def make_fp_mask(actual_gt_box_mask, tp_mask):
    #tp_mask = tf.Print(data=[tf.reduce_sum(tf.cast(tp_mask, dtype=tf.int32))], input_=tp_mask, message="tpmask num nonzeros: ")
    #is an actual box, but does not have an overlap > 0.5 with gt
    fp_mask = tf.logical_and(actual_gt_box_mask, tf.logical_not(tp_mask))
    fp_mask = tf.transpose(fp_mask, perm=(3,0,1,2,4))
    #fp_mask = tf.Print(data=[tf.reduce_sum(tf.cast(fp_mask, dtype=tf.int32))], input_=fp_mask, message="fpmask num nonzeros: ")
    return fp_mask



if __name__ == "__main__":
    import h5py
    with tf.Session() as sess:
        #from dotpy_src.load_data.get_generator import get_generator

        #gen=get_generator("tr", batch_size=2)
        y_true = tf.placeholder(tf.float32,shape=(5,15,5))
        box = h5py.File(configs["tr_data_file"])["boxes"][323:328]
        shapes = [(5, 6, 9, 54),
                  (5, 3, 5, 36),
                    (5, 96, 144, 36),
                    (5, 24, 36, 54),
                    (5, 12, 18, 54),
                    (5, 48, 72, 54),
                    (5, 1, 1, 36)]

        y_preds = [tf.random_normal(mean=0.0,stddev=3.,shape=shape) for shape in shapes]

        da_loss = compute_loss(y_true, y_preds)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/home/evan/hur-detect/src/debug_logs/try1", sess.graph)
        loss_, summary = sess.run([da_loss,merged], feed_dict={y_true:box})
        print loss_
        writer.add_summary(summary,0)
        writer.close()
        



with tf.Session() as sess:
    
    with tf.name_scope("test"):
        a = tf.constant(0)
        tf.summary.scalar("a", a)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/home/evan/hur-detect/src/debug_logs/try2")
    summary = sess.run(merged)
    writer.add_summary(summary,0)
    





