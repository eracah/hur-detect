


import sys

from dotpy_src.config_util import update_configs


kwargs_dict = dict(data_format_args = {'input_shape': (None,16,768,1152),
                    'time_steps_per_example': 1,

                    "time_step_sample_frequency": 2,
                    "time_steps_per_file": 8,
                    "image_variables" :['PRECT','PS','PSL','QREFHT','T200','T500','TMQ','TREFHT',
                                  'TS','U850','UBOT','V850','VBOT','Z1000','Z200','ZBOT'],
                    "label_variables" : ["x_coord","y_coord","w_coord","h_coord","obj","cls"]
                        
                    
    
},


                
                



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


configs = update_configs(configs)





