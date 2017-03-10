


import sys

from dotpy_src.config_util import update_configs

from keras.initializations import he_normal
from dotpy_src.load_data.configs import configs as data_configs



def convert_shape_to_tf(shape):
    #shape is 3 dimensions
    new_shape = (shape[1], shape[2], shape[0])
    return new_shape



configs= {"tensor_input_shape": convert_shape_to_tf(data_configs["input_shape"]),
         #"num_filter_list":[128, 256, 512, 768, 1024], 
        "num_filter_list":[128/8, 256/8, 512/8, 768/8, 1024/8],
         "model": "iclr_semisupervised",
         "num_classes": 4, "w_decay": 0.0005, "w_init":"he_normal" }






configs = update_configs(configs)

