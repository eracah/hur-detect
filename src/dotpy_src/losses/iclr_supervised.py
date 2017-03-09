


from keras.objectives import mean_squared_error



import tensorflow



import tensorflow as tf



import sys



from configs import configs



import numpy as np



from util import *



def evan_sparse_softmax_cross_entropy(y_pred, y_true,depth):
    #y_pred =tf.Print(input_=y_pred,data=[])
    log_sm = log_softmax(y_pred)
    y_true = convert_to_one_hot_encoding(y_true, depth)
    xent = -tf.reduce_sum(y_true * log_sm, axis=-1)
    return xent
     



def convert_to_one_hot_encoding(tens,depth):
    tens=tf.squeeze(tens)
    max_index = tf.reduce_max(tens)

    one_hot= tf.one_hot(indices=tens,depth=depth,axis=-1)
    return one_hot



def log_softmax(y_pred):
    #y_pred = tf.cast(y_pred, tf.float64)
    max_y_pred = tf.reduce_max(y_pred,axis=-1, keep_dims=True)
    max_subtracted = y_pred - max_y_pred
    log_sm = max_subtracted - tf.log(tf.reduce_sum(tf.exp(max_subtracted),axis=-1,keep_dims=True))
    #log_sm= tf.cast(log_sm, tf.float32)
    return log_sm
    
    



def class_loss(y_pred, y_true, obj_tens):    

    y_pred = add_epsilon(y_pred)
    
    losses = evan_sparse_softmax_cross_entropy(y_pred,y_true,depth = configs["num_classes"])
    #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,labels=y_true)
    losses = mask_tens(losses, mask=tf.squeeze(obj_tens))
    mean_loss = average_nonzero_elements(losses)

    return mean_loss
    
    



def obj_loss(y_pred, y_true):
    y_pred = add_epsilon(y_pred)
    
    losses = evan_sparse_softmax_cross_entropy(y_pred,y_true, depth=tf.constant(2))
    loss = tf.reduce_mean(losses)
    num_nonzero = tf.count_nonzero(y_true)
    zeros = lambda: tf.constant(0.0)
    da_loss = lambda: loss
    # if all elements of y_true are zero (aka no objects) then return loss of zero because likely unlabelled frame
    real_loss = tf.case([(tf.greater(num_nonzero,tf.constant(0,dtype=tf.int64)),da_loss)], default=zeros)
    #todo stop gradient if 0
    return real_loss
    



def coord_loss(y_pred, y_true, obj_tens):
#     y_pred = tf.Print(input_=y_pred, data=[tf.reduce_min(y_pred), tf.reduce_max(y_pred)], message="min y_pred, max y_pred: ")
#     y_true = tf.Print(input_=y_true, data=[tf.reduce_min(y_true), tf.reduce_max(y_true)], message="min y_true, max y_true: ")
    losses = smooth_l1(tf.subtract(y_true, y_pred))
    nans = tf.reduce_any(tf.is_nan(losses))
    #losses = tf.Print(input_=losses, data=[nans], message="after smooth l1 nan? ")
    
    mask = tf.concat(values=[obj_tens, obj_tens], concat_dim=3,)
    losses = mask_tens(losses,mask)

    losses = zero_out_nans(losses)
    return tf.reduce_mean(losses)



def reconstruction_loss(y_pred, y_true):
    return tf.reduce_mean(mean_squared_error(y_pred, y_true))



def smooth_l1(x):
   #x = tf.Print(input_=x, data=[tf.reduce_min(x), tf.reduce_max(x)], message="min x, max x: ")
   abs_x = tf.abs(x)
   
   lt_one = tf.less(abs_x, tf.ones_like(abs_x))
   
   
   do_if_lt_one = tf.scalar_mul(x=tf.pow(x,2),scalar=0.5)

   do_if_gt_one = tf.subtract(abs_x,0.5)

   
   
   
   loss = tf.where(lt_one, 
                           do_if_lt_one,
                           do_if_gt_one)

   return loss



def extract_lbl_channels(y_true):
    y_true_xy,y_true_wh,y_true_obj_y_true_cls = tf.split(value=y_true,num_split=3,split_dim=3)
    y_true_obj, y_true_cls = tf.split(value=y_true_obj_y_true_cls, num_split=2, split_dim=3)
    
    y_mask_obj = tf.cast(y_true_obj, tf.float32)
    y_true_obj = tf.squeeze(tf.cast(y_true_obj, tf.int32))
    y_true_cls = tf.squeeze(tf.cast(y_true_cls, tf.int32))
    return y_true_xy,y_true_wh,y_true_obj, y_true_cls, y_mask_obj
    



def extract_pred_channels(y_pred):
    y_pred_xy,y_pred_wh,y_pred_obj, y_pred_cls1,y_pred_cl2 = tf.split(value=y_pred,num_split=5,split_dim=3)
    y_pred_cls = tf.concat(values=[y_pred_cls1,y_pred_cl2], concat_dim=3)
    
    
    return  y_pred_xy,y_pred_wh,y_pred_obj, y_pred_cls



def loss(y_true, y_pred):
    alpha, beta = [configs[k] for k in ["alpha", "beta"]]
    
    y_pred_xy,y_pred_wh,y_pred_obj, y_pred_cls = extract_pred_channels(y_pred)
    
    y_true_xy,y_true_wh,y_true_obj, y_true_cls,y_mask_obj = extract_lbl_channels(y_true)
    
    cls_loss = class_loss(y_pred_cls, y_true_cls,y_mask_obj)
    o_loss = obj_loss(y_pred_obj,y_true_obj)
    
    xy_loss = coord_loss(y_pred_xy, y_true_xy, y_mask_obj)
    
    
    wh_loss = coord_loss(y_pred_wh, y_true_wh, y_mask_obj)
    
    
    
    xy_loss = tf.scalar_mul(alpha, xy_loss)
    #cls_loss = tf.Print(input_=cls_loss,data=[cls_loss, o_loss, xy_loss, wh_loss], message="cls_loss, obj_loss, xy_loss, wh_loss:  ")
    wh_loss = tf.scalar_mul(beta, wh_loss)
    loss = tf.add_n([xy_loss, wh_loss, o_loss, cls_loss])
    #loss = tf.Print(input_=loss, data=[loss], message="final loss: ")
    
    #todo: stop gradient if this is 0
    return loss
    



loss_weights = None



if __name__ == "__main__":
    sess = tf.InteractiveSession()
    brn=tf.contrib.distributions.Bernoulli(p=0.5)
    uni = tf.contrib.distributions.Uniform(0.,1.)
    cat = tf.contrib.distributions.Categorical(p=4*[0.25])
    nrm = tf.contrib.distributions.Normal(mu=0.,sigma=1.)
    
    
    y_true_obj = tf.cast(brn.sample(sample_shape=(5,24,24,1)),tf.float32)
    

    y_pred_cls = uni.sample(sample_shape=(5,24,24, 4))
    y_pred_obj = uni.sample(sample_shape=(5,24,24, 2))
    y_pred_xy = nrm.sample(sample_shape=(5,24,24,2))
    y_pred_wh = nrm.sample(sample_shape=(5,24,24,2))
    y_true_xy = nrm.sample(sample_shape=(5,24,24,2))
    y_true_wh = nrm.sample(sample_shape=(5,24,24,2))
    y_true_cls = tf.cast(cat.sample(sample_shape=(5,24,24,1)), tf.float32)
    
    y_true_im = nrm.sample(sample_shape=(768,768,16))
    y_pred_im = nrm.sample(sample_shape=(768,768,16))
    
    
    outputs = tf.concat(values=[y_pred_xy,y_pred_wh,y_pred_obj, y_pred_cls], concat_dim=3)
    gr_truth = tf.concat(values=[y_true_xy,y_true_wh,y_true_obj, y_true_cls], concat_dim=3)
    loss= sess.run(loss(gr_truth, outputs))
    print loss
    #print sess.run(yolo_semisupervised_loss(outputs, gr_truth))
    
    
    

        



if __name__ == "__main__":
    #sess = tf.InteractiveSession()
    pass
        
        





