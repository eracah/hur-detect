
import matplotlib; matplotlib.use("agg")


import pickle
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
import theano
from theano import tensor as T
import sys
import numpy as np
from copy import deepcopy
#enable importing of notebooks
import inspect
from lasagne.nonlinearities import *
from lasagne.objectives import *
from helper_fxns import get_detec_loss,get_all_boxes, get_MAP
import copy
#if __name__ == "__main__":
    #from data_loader import load_classification_dataset, load_detection_dataset



def get_hyperparams(frame):
    args, _, _, values = inspect.getargvalues(frame)
    #return dict(zip(args,values))
    #del values['frame']
    return values

def build_network(kwargs):
    
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

    
    input_var = T.tensor4('input_var')
    target_var = T.tensor4('target_var') #is of shape (grid_size, grid_size,(n_boxes* 5 + nclass)
    
    print "Building model and compiling functions..." 
    
    #make layers
    networks = build_layers(input_var,kwargs)
    
    #load in any pretrained weights
    if kwargs['load_path'] != "None":
        load_weights(kwargs['load_path'], networks['ae'])
    
    #compile theano functions
    fns = make_fns(networks, input_var, target_var, kwargs)
     

    return fns, networks

def build_layers(input_var, nk):
    '''nk: network_kwargs'''
    '''conv, extra_convs, pool multiple times then fc with dropout, fc with dropout and softmax then reshape'''
    
    '''total number of conv layers is num_convpool * (1 + num_extra_conv)'''
    
    filter_dim = nk['filter_dim']
    base_num_filters = nk['num_filters']
    num_layers = nk['num_layers']
    num_filters = base_num_filters
    

    filters_list = [128, 256, 512, 768, 1024, 1280]
    conv = lasagne.layers.InputLayer(shape=nk['input_shape'])
    for i in range(num_layers):
        if nk["batch_norm"]:
            conv = batch_norm(conv)
        if nk['num_filters'] == 128:
            num_filters = filters_list[i]
        else:
            num_filters = nk['num_filters']
        conv = Conv2DLayer(conv, 
                              num_filters=num_filters, 
                              filter_size=nk['filter_dim'], 
                              pad=nk['filter_dim'] / 2, 
                              stride=2,
                              W=nk['w_init'], 
                              nonlinearity=nk['nonlinearity'])

    encoder = copy.deepcopy(conv)
    for layer in get_all_layers(conv)[::-1]:
        if nk["batch_norm"]:
            conv = batch_norm(conv)
        if isinstance(layer, InputLayer):
            break
        
        conv = InverseLayer(conv, layer)
    
    ae = copy.deepcopy(conv)

    
    coord_net = Conv2DLayer(batch_norm(encoder), num_filters=5, filter_size=1, W=nk['w_init'], nonlinearity=rectify)
    
    class_net = Conv2DLayer(batch_norm(encoder), num_filters=nk['num_classes'], filter_size=1,W=nk['w_init'], nonlinearity=sigmoid)

    bbox_reg = ConcatLayer([coord_net, class_net])
    
    return {'yolo': bbox_reg, 'ae':ae, "encoder":encoder}#, "decoder":decoder_layers}
        

def load_weights(pickle_file_path, network):
    '''grabs weights from an npz file'''
    old_params = pickle.load(open(pickle_file_path, 'r'))

    return set_all_param_values(network, old_params)
    

def make_fns(networks,input_var, target_var, kwargs ):
    '''Compiles theano train, test, box_fns'''
    #deterministic determines whether to use dropout or not in forward pass
    #transpose output to match what loss expects
    for k,v in networks.iteritems():
        kwargs['logger'].info("\n" + k + ": \n")
        for lay in get_all_layers(v):
            kwargs["logger"].info(str(lay) + ", " + str(get_output_shape(lay)))
    
    yolo = networks['yolo']
    ae = networks['ae']
    yolo_test_prediction = lasagne.layers.get_output(yolo, deterministic=True, inputs=input_var)
    yolo_prediction = lasagne.layers.get_output(yolo, deterministic=False,inputs=input_var)
    
    ae_test_prediction = lasagne.layers.get_output(ae, deterministic=True,inputs=input_var)
    ae_prediction = lasagne.layers.get_output(ae, deterministic=False,inputs=input_var)
    
    def make_loss(yolo_pred, ae_pred):
        yolo_loss = make_yolo_loss(yolo_pred)
        ae_loss = make_ae_loss(ae_pred)
        
        #just to make sure we don't compute this if we don't want to
        if kwargs['lambda_ae'] == 0:
            loss = yolo_loss
        else:
            loss = yolo_loss + kwargs['lambda_ae'] * ae_loss
        return loss, yolo_loss, ae_loss
    
    def make_yolo_loss(pred):
        loss = get_detec_loss(pred, target_var, kwargs)
        weightsl2 = lasagne.regularization.regularize_network_params(yolo, lasagne.regularization.l2)
        loss += kwargs['weight_decay'] * weightsl2
        return loss.mean()
    
    def make_ae_loss(pred):
        loss = squared_error(pred, input_var)
        weightsl2 = lasagne.regularization.regularize_network_params(ae, lasagne.regularization.l2)
        loss += kwargs['weight_decay'] * weightsl2
        return loss.mean()
        
    def make_train_fn():
        '''takes as input the input, target vars and ouputs a loss'''
        
        loss, yolo_loss, ae_loss =  make_loss(yolo_prediction, ae_prediction)
        
        #only using params from yolo here -> because decoder has no new params -> tied weights
        params = lasagne.layers.get_all_params(yolo, trainable=True)
        updates = lasagne.updates.adam(loss, params,learning_rate=kwargs['learning_rate'])
#         if kwargs['lambda_ae'] != 0:
#             train_fn = theano.function([input_var, target_var], [loss,yolo_loss, ae_loss], updates=updates)
        train_fn = theano.function([input_var, target_var], [loss, yolo_loss, ae_loss], updates=updates)
        return train_fn
        
    
    def make_test_or_val_fn():
        '''takes as input the input, target vars and ouputs a non-dropout loss and an accuracy (intersection over union)'''
        test_loss, yolo_loss, ae_loss = make_loss(yolo_test_prediction,ae_test_prediction)
        val_fn = theano.function([input_var, target_var], [test_loss,yolo_loss, ae_loss])
        return val_fn
    
    

    
    def make_yolo_pred_fn():
        '''takes as input the input, target vars and outputs the predicted grid'''
        pred_fn = theano.function([input_var], yolo_test_prediction)
        return pred_fn
    
    def make_ae_pred_fn():
        pred_fn = theano.function([input_var], ae_test_prediction)
        return pred_fn
        
    def make_box_fn():
        pred_fn = make_yolo_pred_fn()
        def box_fn(x, y,conf_thresh=kwargs['conf_thresh'], num_classes=kwargs['num_classes']):
            y_tensor = y
            pred_tensor = pred_fn(x)
            pred_boxes, gt_boxes = get_all_boxes(pred_tensor=pred_tensor, 
                                                 y_tensor=y_tensor, 
                                                 conf_thresh=conf_thresh, num_classes=num_classes)
            return pred_boxes, gt_boxes
        return box_fn
            
    def make_map_fn():
        '''takes as input the input, target vars and outputs the predicted and the ground truth boxes)'''
        pred_fn = make_yolo_pred_fn()
        def MAP_fn(inp, gt, conf_thresh=0.7, iou_thresh=0.5,num_classes=4):
            pred = pred_fn(inp)
            MAP = get_MAP(pred,gt, conf_thresh,iou_thresh, num_classes)
            return MAP
    
        return MAP_fn
    
    train_fn = make_train_fn()
    test_or_val_fn = make_test_or_val_fn()
    MAP_fn = make_map_fn()
    yolo_pred_fn = make_yolo_pred_fn()
    ae_pred_fn = make_ae_pred_fn()
    box_fn = make_box_fn()
    
    return {"tr":train_fn, "val":test_or_val_fn, "MAP": MAP_fn, "rec": ae_pred_fn, "box": box_fn}





