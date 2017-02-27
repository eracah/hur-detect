
# coding: utf-8

# In[1]:

import sys
from nbfinder import NotebookFinder
sys.meta_path.append(NotebookFinder())
import keras
from util import make_model_data_struct


# In[2]:

from keras.models import Model


# In[3]:

from keras.layers import Input, Conv2D,Deconv2D,merge
from configs import configs


# In[4]:

encoder_num_filters_list = configs["num_filter_list"]
num_layers = len(encoder_num_filters_list)
inp_shape = configs["input_shape"]


# In[5]:

input_ = Input(inp_shape)


# In[6]:

def encoder(inp):
    x = inp
    for lay_no in range(num_layers):
        num_filters = encoder_num_filters_list[lay_no]
        x = Conv2D(num_filters, 5,5, subsample=(2,2), border_mode="same")(x)

    return x


# In[10]:

def bbox(encoder):
    xy_coords_score = Conv2D(2,3,3,border_mode="same", activation="linear", name="xy_score")(encoder)
    wh_coords_score = Conv2D(2,3,3,border_mode="same", activation="linear", name="wh_score")(encoder)
    objectness_score = Conv2D(2,3,3,border_mode="same", activation="linear", name="objectness_score")(encoder)
    class_score = Conv2D(configs["num_classes"],3,3,border_mode="same", activation="linear", name="class_score")(encoder)
    output = merge([xy_coords_score,wh_coords_score,objectness_score,class_score], concat_axis=-1, mode="concat",name="box_score")
    return output


# In[11]:

output = bbox(encoder(inp=input_))


# In[12]:

def get_model_params():
    return make_model_data_struct(input=input_, output=output)


# In[2]:

# ! jupyter nbconvert --to script iclr_supervised.ipynb

