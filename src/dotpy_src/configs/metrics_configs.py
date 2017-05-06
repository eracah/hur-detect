


import sys
import os 
import numpy as np


kwargs_dict = dict(acc_stuff= { 'select_threshold': 0.01,
                               'select_top_k': 400,
                               'keep_top_k': 200,
                               'nms_threshold': 0.45,
                               'matching_threshold': 0.5})



configs = {}
for kwargs in kwargs_dict.values():
    configs.update(kwargs)


