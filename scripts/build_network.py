
import matplotlib; matplotlib.use("agg")


import lasagne
from lasagne.layers import Conv2DLayer as conv
from lasagne.layers import MaxPool2DLayer as maxpool
from lasagne.layers import dropout
from lasagne.layers import DenseLayer as fully_connected
from lasagne.nonlinearities import rectify as relu
import theano
from theano import tensor as T
import sys
import numpy as np
#enable importing of notebooks
import inspect
from helper_fxns import get_best_box, get_detec_loss, get_iou, make_test_data, get_detec_acc, get_final_box
if __name__ == "__main__":
    from data_loader import load_classification_dataset, load_detection_dataset



def get_hyperparams(frame):
    args, _, _, values = inspect.getargvalues(frame)
    #return dict(zip(args,values))
    #del values['frame']
    return values



def build_network(    input_shape=(None,8,96,96),
                      filter_dim=3,
                      num_convpool=4,
                      num_filters=512,
                      num_fc_units=1024,
                      num_extra_conv=2, 
                      nonlinearity=lasagne.nonlinearities.LeakyRectify(0.1),
                      w_init=lasagne.init.HeUniform(),
                      dropout_p=0.5,      
                      learning_rate = 0.001,
                      weight_decay = 0.0005,
                      momentum = 0.9,
                      delta=0.00001,
                      coord_penalty = 5,
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
    train_fn, val_fn, box_fn = make_fns(network,input_var, target_var, coord_penalty, nonobj_penalty,
                                        learning_rate, momentum, weight_decay, delta)
    
    return train_fn, val_fn, box_fn, network, hyperparams



def build_layers(input_var, **nk):
    '''nk: network_kwargs'''
    '''conv, extra_convs, pool multiple times then fc with dropout, fc with dropout and softmax then reshape'''
    
    '''total number of conv layers is num_convpool * (1 + num_extra_conv)'''
    

    
    
    filter_dim = nk['filter_dim']
    assert filter_dim % 2 != 0, "filter dimensions must be odd to ensure x,y dim preservation in convolutions"
    # convolution parameters that don't change the shape of the input
    conv_kwargs = dict(num_filters=nk['num_filters'], filter_size=(filter_dim,filter_dim), pad=(filter_dim - 1)/2, 
                       nonlinearity=nk['nonlinearity'], W=nk['w_init'])
    
    #shape: 8x8x96
    network = lasagne.layers.InputLayer(shape=nk['input_shape'], input_var=input_var)
    

    #shape: num_filters x 96 * 2^(-num_convpool) x 96 * 2^(-num_convpool)
    for _ in range(nk['num_convpool']):
        network = conv(network,**conv_kwargs )
        
        for _ in range(nk['num_extra_conv']):
            network = conv(network,**conv_kwargs)
        
        network = maxpool(network, pool_size=(2,2))
        
    

    #shape: num_fc_units
    network = fully_connected(dropout(network, p=nk['dropout_p']), num_units=nk['num_fc_units'], 
                              nonlinearity=nk['nonlinearity'])  
    
    
    grid_size, n_boxes, nclass = [nk[k] for k in ['grid_size', 'n_boxes', 'nclass']]
    #shape: (grid_size * grid_size) * (n_boxes* 5 + nclass)  
    network = fully_connected(dropout(network, p=nk['dropout_p']), 
                                      num_units=(grid_size * grid_size) * (n_boxes* 5 + nclass),
                                      nonlinearity=lasagne.nonlinearities.rectify)  
                                      
    
    #shape: grid_size, grid_size,(n_boxes* 5 + nclass))
    network = lasagne.layers.ReshapeLayer(network, shape=([0],grid_size, grid_size,(n_boxes* 5 + nclass)))
                                
    
    return network
        



def load_weights(file_path, network):
    '''grabs weights from an npz file'''
    with np.load(file_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    return network
    



def make_fns(network,input_var, det_target_var, lc, ln, learning_rate, momentum, weight_decay, delta):
    '''Compiles theano train, test, box_fns'''
    #deterministic determines whether to use dropout or not in forward pass
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    prediction = lasagne.layers.get_output(network, deterministic=False)
    
    
    def make_loss(pred):
        loss = get_detec_loss(pred, det_target_var, lc, ln, delta)
        weightsl2 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        loss += weight_decay * weightsl2
        return loss
    
    def make_train_fn():
        '''takes as input the input, target vars and ouputs a loss'''
        
        loss =  make_loss(prediction)
        weightsl2 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, 
                                                    params, 
                                                    learning_rate=learning_rate, 
                                                    momentum=momentum)
        train_fn = theano.function([input_var, det_target_var], loss, updates=updates)
        return train_fn
        
    
    def make_test_or_val_fn():
        '''takes as input the input, target vars and ouputs a non-dropout loss and an accuracy (intersection over union)'''
        test_loss = make_loss(test_prediction)
        test_acc = get_detec_acc(test_prediction, det_target_var)
        val_fn = theano.function([input_var, det_target_var], [test_loss, test_acc])
        return val_fn
    
    
    def make_box_fn():
        '''takes as input the input, target vars and outputs the predicted and the ground truth boxes)'''
        pred_boxes = get_final_box(test_prediction)
        gt_boxes = get_final_box(det_target_var)
        box_fn = theano.function([input_var, det_target_var], [pred_boxes, gt_boxes])
        return box_fn
    
    def make_pred_fn():
        '''takes as input the input, target vars and outputs the predicted grid'''
        pred_fn = theano.function([input_var], test_prediction)
        return pred_fn
        
        
    train_fn = make_train_fn()
    test_or_val_fn = make_test_or_val_fn()
    box_fn = make_box_fn()
    pred_fn = make_pred_fn()
    
    return train_fn, test_or_val_fn, box_fn #,pred_fn



if __name__ == "__main__":
    build_network()









