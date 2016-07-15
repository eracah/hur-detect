
import matplotlib; matplotlib.use("agg")


import lasagne
import theano
from theano import tensor as T
import sys
import numpy as np
#enable importing of notebooks



def build_yolo_layers(input_var,
                      input_shape,
                      num_filters,
                      num_fc_units,
                      num_extra_conv, 
                      num_pool,
                      nonlinearity,
                      w_init,
                      dropout_p):
    # 8x8x96
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    
    #num_filters/2 x 96 x 96 
    network = lasagne.layers.Conv2DLayer(network, num_filters=num_filters / 2, 
                                     filter_size=(3,3),
                                     pad=1,
                                     nonlinearity=nonlinearity,
                                     W=w_init)
    
    for i in range(num_pool - 1):

        #num_filters x 96 / (2*(i+1)) x 96 / (2*(i+1))
        network = lasagne.layers.MaxPool2DLayer(network, 
                                                pool_size=(2,2))

        # num_filters x 96 / (2*(i+1)) x 96 / (2*(i+1))
        network = lasagne.layers.Conv2DLayer(network, num_filters=num_filters, 
                                             filter_size=(3,3),
                                             pad=1,
                                             nonlinearity=nonlinearity,
                                             W=w_init)
        for j in range(num_extra_conv):
            # num_filters x 96 / (2*(i+1)) x 96 / (2*(i+1))
            network = lasagne.layers.Conv2DLayer(network, num_filters=num_filters, 
                                     filter_size=(3,3),
                                     pad=1,
                                     nonlinearity=nonlinearity,
                                     W=w_init)


    
    #1024 x 96 / (2^num_pool) x 96 / (2^num_pool)
    #average pooling
    #name this layer, so we know where it is when we cut layers off this network
    network = lasagne.layers.Pool2DLayer(network, pool_size=(2,2), mode='average_exc_pad', name='avg_pool_layer')
    
    network = lasagne.layers.dropout(network, p=dropout_p) #shape: same as above
    network = lasagne.layers.DenseLayer(
                                lasagne.layers.dropout(network, p=dropout_p),
                                num_units=num_fc_units,
                                nonlinearity=lasagne.nonlinearities.rectify)
    
    network = lasagne.layers.dropout(network, p=dropout_p) #shape: same as above
    network = lasagne.layers.DenseLayer(network, 
                                        num_units=2,
                                        nonlinearity=lasagne.nonlinearities.softmax)
    
    return network



def build_classif_network(learning_rate = 0.01,
                  momentum = 0.9,
                  num_filters=128,
                  num_fc_units=1024,
                  num_extra_conv=0, 
                  num_pool=4,
                  nonlinearity=lasagne.nonlinearities.LeakyRectify(0.1),
                  w_init=lasagne.init.HeUniform(),
                  dropout_p=0.5,
                  weight_decay=0.0005,
                  load=False,
                  load_path='model.npz',
                  input_shape=(None,8,96,96)):
    
    input_var = T.tensor4('input_var')
    classif_target_var = T.ivector('classif_target_var')
    print("Building model and compiling functions...")
    
    
    network = build_yolo_layers(input_var,
                                            input_shape,
                                            num_filters,
                                            num_fc_units,
                                            num_extra_conv, 
                                            num_pool,
                                            nonlinearity,
                                            w_init,
                                            dropout_p)
    
    if load:
        with np.load(load_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network, deterministic=False)
    loss = lasagne.objectives.categorical_crossentropy(prediction, classif_target_var)
    loss = loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step
    params = lasagne.layers.get_all_params(network, trainable=True)
    weightsl2 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    loss += weight_decay * weightsl2
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=momentum)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                classif_target_var)
    test_loss = test_loss.mean()



    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), classif_target_var),
                          dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, classif_target_var], loss, updates=updates)


    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, classif_target_var], [test_loss, test_acc])

    return train_fn, val_fn, input_var, network



if __name__ == "__main__":
    tf,vfn,inp, n = build_classif_network()

