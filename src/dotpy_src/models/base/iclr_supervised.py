


import sys
import keras


from keras.initializations import he_normal

from keras.models import Model

from keras.layers import Input, Conv2D
if __name__ == "__main__":
    sys.path.append("../../../")
from dotpy_src.models.util import make_model_data_struct   
from dotpy_src.configs import configs

from keras.regularizers import l2



encoder_num_filters_list = configs["num_filter_list"]
num_layers = len(encoder_num_filters_list)
inp_shape = configs["tensor_input_shape"]



input_ = Input(inp_shape)



conv_kwargs =  dict(border_mode="same", init=configs["w_init"], W_regularizer=l2(configs["w_decay"]))



def encoder(inp):
    layers = [inp]
    x = inp
    for lay_no in range(num_layers):
        num_filters = encoder_num_filters_list[lay_no]
        x = Conv2D(num_filters, 5,5, subsample=(2,2),activation="relu", name="layer_" + str(lay_no),**conv_kwargs  )(x)
        layers.append(x)
    return layers



def get_base_layers():
    #returns a dictionary mapping the local receptive field size to a layer for all relevant layers
    layers_list = encoder(inp = input_)
    # add one b/c for n layers we want 0th layer up to and including nth layer (1st layer is 2^1 lrf, nth is 2^n)
    lrf_list = [2**i for i in range(num_layers + 1)]
    layers_dict = dict(zip(lrf_list, layers_list))
    return layers_dict








