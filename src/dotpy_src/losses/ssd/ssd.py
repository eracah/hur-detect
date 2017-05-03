


import sys
import tensorflow as tf
slim=tf.contrib.slim
if __name__ == "__main__":
    sys.path.append("../../../")



from dotpy_src.box_encode_decode.ssd.make_anchors_orig import make_anchors_for_one_fmap
from dotpy_src.box_encode_decode.ssd.encode import encode, encode_one_fmap
from dotpy_src.models.configs import configs
from dotpy_src.box_encode_decode.configs import configs as box_configs
from dotpy_src.losses.util import unpack_net_output
from dotpy_src.losses.util import abs_smooth as smooth_L1
configs.update(box_configs)



def compute_loss(y_true, y_preds):
    '''y_true: the boxes Nx15x5 tensor
       y_preds: a list of 7?  tensors of Nxfy x fx x k where k = 4*number of anchors + number_of_anchors*num_classes'''

    losses = [compute_loss_for_one_fmap(y_true, y_pred) for y_pred in y_preds]
    final_loss = tf.add_n(losses)
    return final_loss



def compute_loss_for_one_fmap(y_true, y_pred):
    #TODO 2: make work for batch size greater than 1
    '''y_true: the boxes Nx15x5 tensor
       y_pred: a tensor of Nxfy x fx x k where k = 4*number of anchors + number_of_anchors*num_classes'''
    
    
    feat_shape = tuple([d.value for d in list(y_pred.get_shape()[1:3])])

    gclasses, glocalizations, gscores= encode_one_fmap(y_true, feat_shape=feat_shape)


    localizations, logits = unpack_net_output(y_pred, feat_shape,single_example=False)

        
    final_loss = losses(logits, 
                  localizations,
                  gclasses, 
                  glocalizations, 
                  gscores)
    return final_loss
    



loss_weights = None



def losses(logits, localisations,
           gclasses, glocalisations, gscores,
           match_threshold=0.5,
           negative_ratio=3.,
           alpha=1.,
           label_smoothing=0.,
           scope='ssd_losses'):
    """Define the SSD network losses.
    """
    final_loss = ssd_single_loss(logits, localisations,
                      gclasses, glocalisations, gscores,
                      match_threshold=match_threshold,
                      negative_ratio=negative_ratio,
                      alpha=alpha,
                      label_smoothing=label_smoothing,
                      scope=scope)
    return final_loss


def ssd_single_loss(logits, localisations,
           gclasses, glocalisations, gscores,
           match_threshold=0.5,
           negative_ratio=3.,
           alpha=1.,
           label_smoothing=0.,
           scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: predictions logits Tensor;
      localisations: localisations Tensor;
      gclasses: groundtruth labels Tensor;
      glocalisations: groundtruth localisation Tensor;
      gscores: groundtruth score Tensor;
    """
    dtype = logits.dtype
    # Determine weights Tensor.
    pmask = gscores > match_threshold
    fpmask = tf.cast(pmask, dtype)
    n_positives = tf.reduce_sum(fpmask)

    # Select some random negative entries.
    # n_entries = np.prod(gclasses[i].get_shape().as_list())
    # r_positive = n_positives / n_entries
    # r_negative = negative_ratio * n_positives / (n_entries - n_positives)

    # Negative mask.
    no_classes = tf.cast(pmask, tf.int32)
    predictions = slim.softmax(logits)
    nmask = tf.logical_and(tf.logical_not(pmask),
                           gscores > -0.5)
    fnmask = tf.cast(nmask, dtype)
    nvalues = tf.where(nmask,
                       predictions[:, :, :, :, 0],
                       1. - fnmask)
    nvalues_flat = tf.reshape(nvalues, [-1])
    # Number of negative entries to select.
    n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
    n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)
    n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)
    max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
    n_neg = tf.minimum(n_neg, max_neg_entries)

    val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
    minval = val[-1]
    # Final negative mask.
    nmask = tf.logical_and(nmask, -nvalues > minval)
    fnmask = tf.cast(nmask, dtype)

    # Add cross-entropy loss.

    #cross entropy for positives
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=gclasses)
    loss = tf.losses.compute_weighted_loss(loss, fpmask)
    l_cross_pos = loss

    #cross entropy for negatives
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=no_classes)
    loss = tf.losses.compute_weighted_loss(loss, fnmask)
    l_cross_neg = loss

    

    # Weights Tensor: positive mask + random negative.
    weights = tf.expand_dims(alpha * fpmask, axis=-1)
    loss = smooth_L1(localisations - glocalisations)
    loss = tf.losses.compute_weighted_loss(loss, weights)
    l_loc = loss
    
    final_loss = l_cross_pos + l_cross_neg  + l_loc
    return final_loss



if __name__ == "__main__":
    with tf.Session() as sess:
        from dotpy_src.load_data.get_generator import get_generator

        gen=get_generator("tr", batch_size=4)
        bboxes = tf.placeholder(tf.float32,shape=(4,15,5),name="bboxes")
        shapes = [(4, 6, 9, 48),
                 (4, 3, 5, 32),
                 (4, 96, 144, 32),
                 (4, 24, 36, 48),
                 (4, 12, 18, 48),
                 (4, 48, 72, 48),
                 (4, 1, 1, 32)]

        y_preds = [tf.ones((shape)) for shape in shapes]
        losses = [compute_loss(bboxes,y_preds[i]) for i in range(len(y_preds))]
        final_loss = tf.add_n(losses)
        
        for im, box in gen:
            print sess.run(final_loss, feed_dict={bboxes:box})















# =========================================================================== #
# SSD loss function.
# =========================================================================== #
def ssd_losses(logits, localisations,
           gclasses, glocalisations, gscores,
           match_threshold=0.5,
           negative_ratio=3.,
           alpha=1.,
           label_smoothing=0.,
           scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors;
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
    """
    with tf.name_scope(scope, 'ssd_losses'):
        l_cross_pos = []
        l_cross_neg = []
        l_loc = []
        for i in range(len(logits)):
            dtype = logits[i].dtype
            with tf.name_scope('block_%i' % i):
                # Determine weights Tensor.
                pmask = gscores[i] > match_threshold
                fpmask = tf.cast(pmask, dtype)
                n_positives = tf.reduce_sum(fpmask)

                # Select some random negative entries.
                # n_entries = np.prod(gclasses[i].get_shape().as_list())
                # r_positive = n_positives / n_entries
                # r_negative = negative_ratio * n_positives / (n_entries - n_positives)

                # Negative mask.
                no_classes = tf.cast(pmask, tf.int32)
                predictions = slim.softmax(logits[i])
                nmask = tf.logical_and(tf.logical_not(pmask),
                                       gscores[i] > -0.5)
                fnmask = tf.cast(nmask, dtype)
                nvalues = tf.where(nmask,
                                   predictions[:, :, :, :, 0],
                                   1. - fnmask)
                nvalues_flat = tf.reshape(nvalues, [-1])
                # Number of negative entries to select.
                n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)
                n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)
                max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
                n_neg = tf.minimum(n_neg, max_neg_entries)

                val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                minval = val[-1]
                # Final negative mask.
                nmask = tf.logical_and(nmask, -nvalues > minval)
                fnmask = tf.cast(nmask, dtype)

                # Add cross-entropy loss.
                with tf.name_scope('cross_entropy_pos'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                          labels=gclasses[i])
                    loss = tf.losses.compute_weighted_loss(loss, fpmask)
                    l_cross_pos.append(loss)

                with tf.name_scope('cross_entropy_neg'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                          labels=no_classes)
                    loss = tf.losses.compute_weighted_loss(loss, fnmask)
                    l_cross_neg.append(loss)

                # Add localization loss: smooth L1, L2, ...
                with tf.name_scope('localization'):
                    # Weights Tensor: positive mask + random negative.
                    weights = tf.expand_dims(alpha * fpmask, axis=-1)
                    loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
                    loss = tf.losses.compute_weighted_loss(loss, weights)
                    l_loc.append(loss)

        # Additional total losses...
        with tf.name_scope('total'):
            total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
            total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
            total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
            total_loc = tf.add_n(l_loc, 'localization')

            # Add to EXTRA LOSSES TF.collection
            tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
            tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
            tf.add_to_collection('EXTRA_LOSSES', total_cross)
            tf.add_to_collection('EXTRA_LOSSES', total_loc)









