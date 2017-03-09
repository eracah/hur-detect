


import sys

from dotpy_src.config_util import update_configs

scale_factor = 32
configs= {"scale_factor": scale_factor,
         "anchor_box_size": scale_factor,
         "alpha":1., "beta":1., "autoencoder_weight": 0.1, "num_classes":4}



configs = update_configs(configs)

