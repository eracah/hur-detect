


import sys
import os 



kwargs_dict = dict(data_format_args = {'raw_input_shape': (16,768,1152), "input_shape":(16,768,1152)},
                                       
                                       
#                                        , "label_shape": (6,24,24) },

                label_kwargs = {"scale_factor": 32, "one_hot_labels": False, "tf_format":True, "num_classes": 4,
                                "label_maker_name": "yolo", "num_max_boxes":15},

                tr_val_test_args = {'batch_size' : 2,
                                    "num_tr_ims": 4,
                                    "num_val_ims": 4,
                                    "num_test_ims": 32
                                   },

                file_args = {'labels_file': "/home/evan/data/climate/labels/labels.csv",
                             'data_file': "/home/evan/data/climate/climo_1980.h5"},

                data_type = {"data_name": "climate"},
                   


                  
                  )
    


configs = {}
for kwargs in kwargs_dict.values():
    configs.update(kwargs)






