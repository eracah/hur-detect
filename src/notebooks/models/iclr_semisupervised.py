
# coding: utf-8

# In[1]:

import sys
from nbfinder import NotebookFinder
sys.meta_path.append(NotebookFinder())
import keras


# In[2]:

from keras.models import Model


# In[1]:

from keras.layers import Input, Conv2D,Deconv2D
from configs import configs
from util import make_model_data_struct


# In[4]:

from iclr_supervised import encoder, bbox


# In[5]:

encoder_num_filters_list = configs["num_filter_list"]
num_layers = len(encoder_num_filters_list)
inp_shape = configs["input_shape"]


# In[6]:

input_ = Input(inp_shape)


# In[7]:

inp_shape


# In[8]:

def decoder(code_layer_output):
    # us all filter sizes from encoder except for last, most recent one
    decoder_num_filters_list = encoder_num_filters_list[:-1]
    # use them in reverse
    decoder_num_filters_list.reverse()
    
    # so the last layer gets to shape of input, last layer num_filters == input num channels
    decoder_num_filters_list.append(inp_shape[0])

    x = code_layer_output
    
    for lay_no in range(num_layers):
        num_filters = decoder_num_filters_list[lay_no]
        o_shape = [None, num_filters] + [dim / 2**(num_layers - lay_no -1) for dim in inp_shape[1:3]]
        print o_shape
        if lay_no == num_layers - 1:
            name = "reconstruction"
        else:
            name = "deconv_" + str(lay_no)
        x = Deconv2D(num_filters, 5,5,subsample=(2,2), output_shape=o_shape,  border_mode="same", name=name)(x)


    return x


# In[9]:

code_layer_output = encoder(inp=input_)


# In[10]:

reconstruction = decoder(code_layer_output)


# In[13]:

box_score= bbox(code_layer_output)


# In[14]:

def get_model_params():
    return make_model_data_struct(input=input_, output = [box_score, reconstruction])


# In[2]:

#! jupyter nbconvert --to script iclr_semisupervised.ipynb


# In[ ]:



