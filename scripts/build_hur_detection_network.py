
import matplotlib; matplotlib.use("agg")


import lasagne
import theano
from theano import tensor as T
import sys
import numpy as np
#enable importing of notebooks
import inspect
from detec_helper_fxns import get_best_box, get_detec_loss, get_iou, make_test_data, get_detec_acc, get_final_box
if __name__ == "__main__":
    from data_loader import load_classification_dataset, load_detection_dataset



def get_hyperparams(frame):
    args, _, _, values = inspect.getargvalues(frame)
    #return dict(zip(args,values))
    #del values['frame']
    return values



def build_det_layers(class_net, layers_to_remove,
                                  num_filters,
                                  num_fc_units,
                                  num_extra_conv, 
                                  nonlinearity,
                                  n_boxes,
                                  nclass,
                                  grid_size,
                                  w_init,
                                  dropout_p):
    
    #define some syntatic sugar
    conv = lasagne.layers.Conv2DLayer
    fc = lasagne.layers.DenseLayer
    conv_kwargs = dict(num_filters=num_filters, filter_size=(3,3), pad=1, nonlinearity=nonlinearity, W=w_init)
    
    #remove the fc, softmax, avg pooling layers
    class_net = strip_off_classif_fc_layers(class_net)

        
    #num_filters x 96 / (2^num_pool) x 96 / (2^num_pool)
    network = conv(class_net, **conv_kwargs)
    
    
    #shape: num_filters x 96 / (2^num_pool) x 96 / (2^num_pool)
    for i in range(num_extra_conv):
        network = conv(network, **conv_kwargs) 
        
    
    network = lasagne.layers.dropout(network, p=dropout_p) #shape: same as above
    network = fc(network, num_units=num_fc_units, nonlinearity=nonlinearity)  #shape: num_fc_units
    network = lasagne.layers.dropout(network, p=dropout_p) #shape: same as above
    network = fc(network, num_units=(grid_size * grid_size) * (n_boxes* 5 + nclass),
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
                      layers_to_remove=3,
                      delta=0.00001,
                      num_filters=512,
                      num_fc_units=1024,
                      num_extra_conv=1, 
                      nonlinearity=lasagne.nonlinearities.LeakyRectify(0.1),
                      n_boxes=1,
                      nclass=1,
                      grid_size=6,
                      w_init=lasagne.init.HeUniform(),
                      dropout_p=0.5,
                      lc = 5, #penalty for getting coordinates wrong
                      ln = 0.5, #penalty for guessing object when there isnt one
                      learning_rate = 0.001,
                      weight_decay = 0.0005,
                      momentum = 0.9,
                      load=False,
                      load_path=None):
    

    hyperparams = get_hyperparams(inspect.currentframe())
    #define target_var
    det_target_var = T.tensor4('det_target_var') #is of shape (grid_size, grid_size,(n_boxes* 5 + nclass)
    
    print "Building model and compiling functions..." 
    
    #make layers
    network = build_det_layers(class_net, layers_to_remove,
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
    from build_hur_classif_network import build_classif_network


    _,_,inp,cn = build_classif_network(dropout_p=0, input_shape=(None,1,96,96),num_filters=20, num_fc_units=50 )

    train_fn, val_fn, box_fn, network, hyperparams = build_det_network(cn, inp, num_filters=20, num_fc_units=50)

    x_tr, grid_tr,x_te,grid_te,x_v,grid_v = load_detection_dataset(num_ims=20)



if __name__ == "__main__":
    from build_hur_classif_network import build_classif_network


    _,_,inp,cn = build_classif_network(dropout_p=0, input_shape=(None,8,96,96) )

    train_fn, val_fn,box_fn,network = build_det_network(cn, inp)

   # x_tr, grid_tr,x_te,grid_te,x_v,grid_v = load_detection_dataset(num_ims=20)
    








