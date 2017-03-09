


import sys

from dotpy_src.config_util import update_configs

from keras.initializations import he_normal



configs= {"input_shape": (768,768,16),
         #"num_filter_list":[128, 256, 512, 768, 1024], 
        "num_filter_list":[128/8, 256/8, 512/8, 768/8, 1024/8],
         "model": "iclr_semisupervised",
         "num_classes": 4, "w_decay": 0.0005, "w_init":"he_normal" }






configs = update_configs(configs)

