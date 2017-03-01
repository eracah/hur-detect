
# coding: utf-8

# In[36]:

import sys
from nbfinder import NotebookFinder
sys.meta_path.append(NotebookFinder())
import keras
from util import make_model_data_struct


# In[37]:

from keras.initializations import he_normal


# In[38]:

from keras.models import Model


# In[39]:

from keras.layers import Input, Conv2D,Deconv2D,merge, BatchNormalization,Activation
from configs import configs
from keras.layers.advanced_activations import LeakyReLU


# In[40]:

from keras.regularizers import l2


# In[41]:

encoder_num_filters_list = configs["num_filter_list"]
num_layers = len(encoder_num_filters_list)
inp_shape = configs["input_shape"]


# In[42]:

input_ = Input(inp_shape)


# In[43]:

conv_kwargs =  dict(border_mode="same", init=configs["w_init"], W_regularizer=l2(configs["w_decay"]))


# In[44]:

def encoder(inp):
    x = inp
    for lay_no in range(num_layers):
        num_filters = encoder_num_filters_list[lay_no]
        conv = Conv2D(num_filters, 5,5, subsample=(2,2),**conv_kwargs )(x)
        bn = BatchNormalization()(conv)
        x = Activation(activation=LeakyReLU(alpha=0.1))(bn)

    return x


# In[45]:

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


# In[46]:

output = bbox(encoder(inp=input_))


# In[47]:

def get_model_params():
    return make_model_data_struct(input=input_, output=output)


# In[49]:

get_ipython().system(u' jupyter nbconvert --to script iclr_supervised.ipynb')


# In[ ]:



