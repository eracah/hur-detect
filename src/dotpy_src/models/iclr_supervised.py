


import sys
import keras
from util import make_model_data_struct



from keras.initializations import he_normal



from keras.models import Model



from keras.layers import Input, Conv2D,Deconv2D,merge, BatchNormalization,Activation
from configs import configs
from keras.layers.advanced_activations import LeakyReLU



from keras.regularizers import l2



encoder_num_filters_list = configs["num_filter_list"]
num_layers = len(encoder_num_filters_list)
inp_shape = configs["tensor_input_shape"]



input_ = Input(inp_shape)



conv_kwargs =  dict(border_mode="same", init=configs["w_init"], W_regularizer=l2(configs["w_decay"]))



def encoder(inp):
    x = inp
    for lay_no in range(num_layers):
        num_filters = encoder_num_filters_list[lay_no]
        x = Conv2D(num_filters, 5,5, subsample=(2,2),activation="relu",**conv_kwargs  )(x)

    return x



def bbox(encoder):
    
    xy_coords_score = Conv2D(2,3,3,
                             activation="linear", 
                             name="xy_score",**conv_kwargs)(encoder)
    
    
    
    wh_coords_score = Conv2D(2,3,3,
                             activation="linear",
                             name="wh_score",**conv_kwargs)(encoder)
    
    objectness_score = Conv2D(2,3,3,
                              activation="relu", name="objectness_score",**conv_kwargs)(encoder)
    
    class_score = Conv2D(configs["num_classes"],3,3,
                         activation="relu", 
                         name="class_score", **conv_kwargs)(encoder)
    
    
    output = merge([xy_coords_score,wh_coords_score,objectness_score,class_score],
                   concat_axis=-1, mode="concat",name="box_score")
    return output



output = bbox(encoder(inp=input_))



def get_model_params():
    return make_model_data_struct(input=input_, output=output)








