
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



def build_network(mode='detection', network_kwargs={}, detec_specific_kwargs={},
                  pretrained_class_network=None, classif_weight_load_path=None, detect_weight_load_path=None):
        
    '''just classification'''
    if mode=='classification':
        pass
        #train_fn, val_fn, input_var, network = build_classif_network(**network_kwargs)
        
   
    elif mode == 'detection':
        '''detection with an in-memory classification network'''
        if pretrained_class_network:
            train_fn,val_fn, box_fn, network, hyperparams = build_det_network(inmem_class_network,
                                                                              input_var,
                                                                              **network_kwargs)
        elif classif_weight_load_path:
            _,_,input_var,class_network = build_classif_network(load=True, load_path=classif_weight_load_path, 
                                                                **network_kwargs)
            network_kwargs.update(detec_specific_kwargs)
            train_fn,val_fn, box_fn, network,hyperparams = build_det_network(class_network,
                                                                             input_var, 
                                                                             **network_kwargs)
        elif detect_weight_load_path:
            network_kwargs.update(detec_specific_kwargs)
            train_fn,
            val_fn, 
            box_fn, 
            network, 
            hyperparams = build_det_network(class_network,input_var,load_path=detect_weight_load_path,**network_kwargs)

        else:
            print "running on non pretrained classif network!"
            train_fn,val_fn,input_var,class_network = build_classif_network(**network_kwargs)
            
            network_kwargs.update(detec_specific_kwargs)
            train_fn,val_fn,box_fn, network, hyperparams = build_det_network(class_network,
                                                                             input_var, 
                                                                             **network_kwargs)
        
        return train_fn, val_fn, box_fn, network, hyperparams



def build_det_layers(class_net,
                                  num_filters,
                                  num_fc_units,
                                  num_extra_conv, 
                                  nonlinearity,
                                  n_boxes,
                                  nclass,
                                  grid_size,
                                  w_init,
                                  dropout_p):
    
    '''Takes a pretrained classification net and adds a few convolutional layers on top of it'''
    
    #define some syntatic sugar
    conv_kwargs = dict(num_filters=num_filters, filter_size=(3,3), pad=1, nonlinearity=nonlinearity, W=w_init)
    
    #remove the fc, softmax, avg pooling layers from the classification network
    class_net = strip_off_classif_fc_layers(class_net)

        
    #num_filters x 96 / (2^num_pool) x 96 / (2^num_pool)
    network = conv(class_net, **conv_kwargs)
    
    
    #shape: num_filters x 96 / (2^num_pool) x 96 / (2^num_pool)
    for i in range(num_extra_conv):
        network = conv(network, **conv_kwargs) 
        
    
    network = dropout(network, p=dropout_p) #shape: same as above
    network = fully_connected(network, num_units=num_fc_units, nonlinearity=nonlinearity)  #shape: num_fc_units
    network = dropout(network, p=dropout_p) #shape: same as above
    network = fully_connected(network, num_units=(grid_size * grid_size) * (n_boxes* 5 + nclass),
                                    nonlinearity=lasagne.nonlinearities.rectify)  
                                    #shape: (grid_size * grid_size) * (n_boxes* 5 + nclass)     
    network = lasagne.layers.ReshapeLayer(network, shape=([0],grid_size, grid_size,(n_boxes* 5 + nclass)))
                                    #shape: grid_size, grid_size,(n_boxes* 5 + nclass))
    
    return network
    

def strip_off_classif_fc_layers(class_net):
    while class_net.name != 'avg_pool_layer':
        #keep cutting off layers until you get to the avg pool layer
        class_net = class_net.input_layer
    #then cut off the avg pool later
    class_net = class_net.input_layer
    return class_net
    



def build_det_network(class_net, input_var,
                      delta=0.00001,
                      num_filters=512,
                      num_fc_units=1024,
                      num_extra_conv=1, 
                      nonlinearity=lasagne.nonlinearities.LeakyRectify(0.1),
                     
                      w_init=lasagne.init.HeUniform(),
                      dropout_p=0.5,

                      learning_rate = 0.001,
                      weight_decay = 0.0005,
                      momentum = 0.9,
                      lc = 5, #penalty for getting coordinates wrong
                      ln = 0.5, #penalty for guessing object when there isnt one
                      n_boxes=1,
                      nclass=1,
                      grid_size=6,
                      load=False,
                      load_path=None):
    
    '''Takes a pretrained classification net and adds a few convolutional layers on top of it
    and defines a detection loss function'''
    '''Args:
                      delta: smoothing constant to loss function (ie sqrt(x + delta)) 
                            -> if x is 0 gradient is undefined
                      num_filters
                      num_fc_units
                      num_extra_conv: conv layers to add on to classification network 
                      nonlinearity: which nonlinearity to use throughout
                      n_boxes: how many boxes should be predicted at each grid point,
                      nclass: how many classes are we predicting,
                      grid_size: size of the grid that encodes various 
                                locations of image (ie in the YOLO paper they use 7x7 grid)
                      w_init: weight intitialization
                      dropout_p: prob of dropping unit
                      lc : penalty in YOLO loss function for getting coordinates wrong
                      ln: penalty in YOLO loss for guessing object when there isn't one
                      learning_rate
                      weight_decay
                      momentum
                      load: whether to load weights or not
                      load_path: path for loading weights'''

    hyperparams = get_hyperparams(inspect.currentframe())
    
    #define target_var
    det_target_var = T.tensor4('det_target_var') #is of shape (grid_size, grid_size,(n_boxes* 5 + nclass)
    
    print "Building model and compiling functions..." 
    
    #make layers
    network = build_det_layers(class_net,
                                  num_filters,
                                  num_fc_units,
                                  num_extra_conv, 
                                  nonlinearity,
                                  n_boxes,
                                  nclass,
                                  grid_size,
                                  w_init,
                                  dropout_p)
    
    #load in any pretrained weights
    if load:
        network = load_weights(load_path, network)
    
    #compile theano functions
    train_fn, val_fn, box_fn = make_fns(network,input_var, det_target_var, lc, ln,
                                        learning_rate, momentum, weight_decay, delta)
    
    return train_fn, val_fn, box_fn, network, hyperparams



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



def build_classif_layers(input_var,
                      input_shape,
                      num_filters,
                      num_fc_units,
                      num_extra_conv, 
                      num_pool,
                      nonlinearity,
                      w_init,
                      dropout_p):
    
    '''builds architecture for classification'''
    
    conv_kwargs = dict(num_filters=num_filters, filter_size=(3,3), pad=1,nonlinearity=nonlinearity, W=w_init)
    
    ''' 8x8x96'''
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    
    '''num_filters x 96 x 96 '''
    network = conv(network, **conv_kwargs)
    
    for i in range(num_pool):

        '''num_filters x 96 / (2^i) x 96 / (2^i)'''
        network = maxpool(network, pool_size=(2,2))

        '''num_filters x 96 / (2^i) x 96 / (2^i)'''
        network = conv(network,**conv_kwargs )
        
        for j in range(num_extra_conv):
            
            '''num_filters x 96 / (2^i) x 96 / (2^i)'''
            network = conv(network,**conv_kwargs)


    
    '''shape: num_filters x 96 / (2^num_pool) x 96 / (2^num_pool)'''
   
    #average pooling
    
    '''name this layer, so we know where it is when we cut layers off this network'''
    network = lasagne.layers.Pool2DLayer(network, pool_size=(2,2), mode='average_exc_pad', name='avg_pool_layer')
    
    network = dropout(network, p=dropout_p) #shape: same as above
    network = fully_connected(network,num_units=num_fc_units, nonlinearity=relu) #shape: num_fc_units
    
    network = dropout(network, p=dropout_p) #shape: same as above
    network = fully_connected(network, num_units=2, nonlinearity=lasagne.nonlinearities.softmax) #shape: 2 (2 classes)
    
    return network

def build_classif_network(learning_rate = 0.01,
                  momentum = 0.9,
                  num_filters=128,
                  num_fc_units=1024,
                  num_extra_conv=0, 
                  num_pool=3,
                  nonlinearity=lasagne.nonlinearities.LeakyRectify(0.1),
                  w_init=lasagne.init.HeUniform(),
                  dropout_p=0.5,
                  weight_decay=0.0005,
                  load=False,
                  load_path='model.npz',
                  input_shape=(None,8,96,96)):
    
    ''' builds network for classification, which is pretrained on a classification task
        (ie: first we put in images and have the network guess if its a hurricane or not be fire diubgf)'''
    '''Args:
                  learning_rate
                  momentum
                  num_filters: number of convolutional filters
                  num_fc_units: number of units out of fc layer
                  num_extra_conv: number of additional conv layers before avg pooling and fc
                  num_pool: number of max pool layers (determines number of matching conv layers as well)
                  nonlinearity=lasagne.nonlinearities.LeakyRectify(0.1)
                  w_init: weight intialization strategy
                  dropout_p: probabiltiy of setting units to zero in dropout scheme
                  weight_decay: coefficient to L2 norm weight penalty
                  load: whether to load weights
                  load_path: path of where file of weights is
                  input_shape: input image dimensions
                  '''
    
    
    input_var = T.tensor4('input_var')
    classif_target_var = T.ivector('classif_target_var')
    
    print("Building model and compiling functions...")
    
    
    '''get actual architecture'''
    network = build_classif_layers(input_var,
                                            input_shape,
                                            num_filters,
                                            num_fc_units,
                                            num_extra_conv, 
                                            num_pool,
                                            nonlinearity,
                                            w_init,
                                            dropout_p)
    
    '''load weights if necessary'''
    if load:
        with np.load(load_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)

    
    '''calculate loss -> standard cross entropy with weight decay'''
    prediction = lasagne.layers.get_output(network, deterministic=False)
    loss = lasagne.objectives.categorical_crossentropy(prediction, classif_target_var)
    loss = loss.mean()
    weightsl2 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    loss += weight_decay * weightsl2
    
    
    '''calculate test loss (cross entropy with no regularization) and accuracy'''
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                classif_target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), classif_target_var),
                          dtype=theano.config.floatX)

    '''calculate updates -> nesterov momentum sgd'''
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=momentum)




    '''train_fn -> takes in input,label pairs -> outputs loss '''
    train_fn = theano.function([input_var, classif_target_var], loss, updates=updates)


    '''val_fn -> takes in input,label pairs -> outputs non regularized loss and accuracy '''
    val_fn = theano.function([input_var, classif_target_var], [test_loss, test_acc])

    return train_fn, val_fn, input_var, network



if __name__ == "__main__":
    build_network()









