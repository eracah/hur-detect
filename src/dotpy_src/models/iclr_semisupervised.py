


import sys
import keras



from keras.models import Model



from keras.layers import Input, Conv2D,Deconv2D
from configs import configs
from util import make_model_data_struct



from keras.regularizers import l2



from iclr_supervised import encoder, bbox



encoder_num_filters_list = configs["num_filter_list"]
num_layers = len(encoder_num_filters_list)
inp_shape = configs["input_shape"]



input_ = Input(inp_shape)



conv_kwargs =  dict(border_mode="same", init=configs["w_init"], W_regularizer=l2(configs["w_decay"]))



def decoder(code_layer_output):
    # us all filter sizes from encoder except for last, most recent one
    decoder_num_filters_list = encoder_num_filters_list[:-1]
    # use them in reverse
    decoder_num_filters_list.reverse()
    
    # so the last layer gets to shape of input, last layer num_filters == input num channels
    decoder_num_filters_list.append(inp_shape[-1])

    x = code_layer_output
    
    for lay_no in range(num_layers):
        num_filters = decoder_num_filters_list[lay_no]
        o_shape = [None]+ [dim / 2**(num_layers - lay_no -1) for dim in inp_shape[0:2]] + [num_filters]
        #print o_shape
        if lay_no == num_layers - 1:
            name = "reconstruction"
        else:
            name = "deconv_" + str(lay_no)
        x = Deconv2D(num_filters, 5,5,subsample=(2,2), output_shape=o_shape,  name=name, **conv_kwargs)(x)


    return x



code_layer_output = encoder(inp=input_)



reconstruction = decoder(code_layer_output)



box_score= bbox(code_layer_output)



def get_model_params():
    return make_model_data_struct(input=input_, output = [box_score, reconstruction])








