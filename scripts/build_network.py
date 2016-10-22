
import matplotlib; matplotlib.use("agg")


import lasagne
from lasagne.layers import *

from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import dropout
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify
import theano
from theano import tensor as T
import sys
import numpy as np
from copy import deepcopy
#enable importing of notebooks
import inspect
from lasagne.nonlinearities import *
from lasagne.objectives import *
from helper_fxns import get_detec_loss, get_boxes_ap
#if __name__ == "__main__":
    #from data_loader import load_classification_dataset, load_detection_dataset

def get_hyperparams(frame):
    args, _, _, values = inspect.getargvalues(frame)
    #return dict(zip(args,values))
    #del values['frame']
    return values

def build_network(    input_shape=(1,16,768,1152),
                      num_classes = 4,
                      filter_dim=3,
                      num_convpool=4,
                      scale_factor=64,
                      num_filters=128,
                      num_layers = 6,
                      num_fc_units=1024,
                      num_extra_conv=2, 
                      nonlinearity=lasagne.nonlinearities.LeakyRectify(0.1),
                      w_init=lasagne.init.HeUniform(),
                      dropout_p=0.5,      
                      learning_rate = 0.00001,
                      weight_decay = 0.0005,
                      momentum = 0.9,
                      delta=0.00001,
                      coord_penalty = 5,
                      size_penalty = 5,
                      nonobj_penalty = 0.5,
                      n_boxes=1,
                      nclass=1,
                      grid_size=6,
                      load=False,
                      load_path=None):
    
    '''Takes a pretrained classification net and adds a few convolutional layers on top of it
    and defines a detection loss function'''
    '''Args:
                      
                      num_convpool: number of conv layer-pool layer pairs
                      delta: smoothing constant to loss function (ie sqrt(x + delta)) 
                            -> if x is 0 gradient is undefined
                      num_filters
                      num_fc_units
                      num_extra_conv: conv layers to add on to each conv layer before max pooling
                      nonlinearity: which nonlinearity to use throughout
                      n_boxes: how many boxes should be predicted at each grid point,
                      nclass: how many classes are we predicting,
                      grid_size: size of the grid that encodes various 
                                locations of image (ie in the YOLO paper they use 7x7 grid)
                      w_init: weight intitialization
                      dropout_p: prob of dropping unit
                      coord_penalty : penalty in YOLO loss function for getting coordinates wrong
                      nonobj_penalty: penalty in YOLO loss for guessing object when there isn't one
                      learning_rate
                      weight_decay
                      momentum
                      load: whether to load weights or not
                      load_path: path for loading weights'''

    #get all key,value args from function
    hyperparams = get_hyperparams(inspect.currentframe())
    
    input_var = T.tensor4('input_var')
    target_var = T.tensor4('target_var') #is of shape (grid_size, grid_size,(n_boxes* 5 + nclass)
    
    print "Building model and compiling functions..." 
    
    #make layers
    network = build_layers(input_var, **hyperparams)
    
    #load in any pretrained weights
    if load:
        network = load_weights(load_path, network)
    
    #compile theano functions
    train_fn, val_fn,pred_fn, ap_box_fn = make_fns(network,input_var, target_var, coord_penalty, size_penalty, nonobj_penalty,
                                        learning_rate, momentum, weight_decay, delta,scale_factor)
    
    #box_fn
    return train_fn, val_fn,pred_fn, ap_box_fn, network, hyperparams

def build_layers(input_var, **nk):
    '''nk: network_kwargs'''
    '''conv, extra_convs, pool multiple times then fc with dropout, fc with dropout and softmax then reshape'''
    
    '''total number of conv layers is num_convpool * (1 + num_extra_conv)'''
    
    filter_dim = nk['filter_dim']
    base_num_filters = nk['num_filters']
    num_layers = nk['num_layers']
    num_filters = base_num_filters
    
    network = lasagne.layers.InputLayer(shape=nk['input_shape'], input_var=input_var)
    
    for _ in range(num_layers):
        network = Conv2DLayer(batch_norm(network), 
                              num_filters=num_filters, 
                              filter_size=nk['filter_dim'], 
                              pad=nk['filter_dim'] / 2, stride=2, W=nk['w_init'], nonlinearity=nk['nonlinearity'])
        num_filters *= 2

    
    coord_net = Conv2DLayer(batch_norm(network), num_filters=5, filter_size=1,W=nk['w_init'], nonlinearity=rectify)
    
    class_net = Conv2DLayer(batch_norm(network), num_filters=nk['num_classes'], filter_size=1,W=nk['w_init'], nonlinearity=sigmoid)

    network = ConcatLayer([coord_net, class_net])
    
    return network
        

def load_weights(file_path, network):
    '''grabs weights from an npz file'''
    with np.load(file_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    return network
    

def make_fns(network,input_var, det_target_var, lcxy, lchw, ln, learning_rate, momentum, weight_decay, delta,scale_factor):
    '''Compiles theano train, test, box_fns'''
    #deterministic determines whether to use dropout or not in forward pass
    #transpose output to match what loss expects
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    prediction = lasagne.layers.get_output(network, deterministic=False)
    
    
    def make_loss(pred):
        loss = get_detec_loss(pred, det_target_var, lcxy, lchw, ln, delta)
        weightsl2 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        loss += weight_decay * weightsl2
        return loss.mean()
    
    def make_train_fn():
        '''takes as input the input, target vars and ouputs a loss'''
        
        loss=  make_loss(prediction)
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.adam(loss, params,learning_rate=learning_rate)
        train_fn = theano.function([input_var, det_target_var], loss, updates=updates)
        return train_fn
        
    
    def make_test_or_val_fn():
        '''takes as input the input, target vars and ouputs a non-dropout loss and an accuracy (intersection over union)'''
        test_loss = make_loss(test_prediction)
        val_fn = theano.function([input_var, det_target_var], test_loss)
        return val_fn
    
    

    
    def make_pred_fn():
        '''takes as input the input, target vars and outputs the predicted grid'''
        pred_fn = theano.function([input_var], test_prediction)
        return pred_fn
        
    def make_ap_box_fn():
        '''takes as input the input, target vars and outputs the predicted and the ground truth boxes)'''
        pred_fn = make_pred_fn()
        def ap_box_fn(inp,gt,conf_thresh=0.7,iou_thresh=0.5):
            pred = pred_fn(inp)
            ap, pred_boxes, gt_boxes = get_boxes_ap(pred,gt, conf_thresh,iou_thresh)
            return ap, pred_boxes, gt_boxes
    
        return ap_box_fn
    
    train_fn = make_train_fn()
    test_or_val_fn = make_test_or_val_fn()
    ap_box_fn = make_ap_box_fn()
    pred_fn = make_pred_fn()
    
    return train_fn, test_or_val_fn,pred_fn, ap_box_fn

if __name__ == "__main__":
    train_fn, val_fn,pred_fn, ap_box_fn, network, hyperparams = build_network(num_filters=2)

    from netcdf_loader import bbox_iterator

    for e in range(10):
        for x,y in bbox_iterator(years=[1979], days=3,data_dir="/storeSSD/eracah/data/netcdf_ims/", metadata_dir="/storeSSD/eracah/data/metadata/"):
            x = np.squeeze(x,axis=2)
            y = np.squeeze(y,axis=1)
            #pred = pred_fn(x)
            #print pred
            #print pred.shape
            loss = train_fn(x,y)
            print loss
            #vloss, acc = val_fn(x,y)
            ap, pred_box, gt_box =  ap_box_fn(x,y, conf_thresh=0.5,iou_thresh=0.1)
            print ap
    #         print loss
    #         print vloss
    #         print acc
            #print pred_box
            #print gt_box
  




