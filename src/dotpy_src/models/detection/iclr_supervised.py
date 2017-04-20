


import sys
import keras


from keras.initializations import he_normal

from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Conv2D,merge
import importlib



if __name__ == "__main__":
    sys.path.append("../../../")
from dotpy_src.models.configs import configs
from dotpy_src.models.util import make_model_data_struct
from dotpy_src.models.base.get_base_model import get_base_model_layers                                  



conv_kwargs =  dict(border_mode="same", init=configs["w_init"], W_regularizer=l2(configs["w_decay"]))



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



# layers is a dict matching local receptive field to layer
layers = get_base_model_layers()



# encoder is last layer, so layer with largest receptive field
encoder_layer = layers[max(layers.keys())]

#input tensor has local recpetive field of 1 (layers is a dict NOT a list)
input_tensor = layers[1]



output = bbox(encoder_layer)



def get_model_params():
    return make_model_data_struct(input=input_tensor, output=output)

