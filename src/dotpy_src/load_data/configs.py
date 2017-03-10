


import sys
import os 
if os.path.basename(os.getcwd()) == "src":
    from dotpy_src.config_util import update_configs


kwargs_dict = dict(data_format_args = {'input_shape': (16,768,768), "label_shape": (6,24,24) },


                
                



tr_val_test_args = {'batch_size' : 16,
                    "num_tr_ims": 48,
                    "num_val_ims": 16,
                    "num_test_ims": 32
                   },

file_args = {'labels_file': "/home/evan/data/climate/labels/labels.csv",
             'data_dir': "/home/evan/data/climate/input",
             "data_list_dir": "/home/evan/data/climate/lists/small-gpu-size_lists/semisupervised/"},
             
data_type = {"data_name": "climate"})
    


configs = {}
for kwargs in kwargs_dict.values():
    configs.update(kwargs)

if os.path.basename(os.getcwd()) == "src":
    configs = update_configs(configs)

