


import sys
import os 



kwargs_dict = dict(data_format_args = {'raw_input_shape': (16,768,1152), "input_shape":(16,768,1152)},
                                       
                                       
#                                        , "label_shape": (6,24,24) },

                label_kwargs = {"tf_format":True, "num_classes": 4, "num_max_boxes":15},

                tr_val_test_args = {'batch_size' : 2,
                                    "num_tr_ims": 4,
                                    "num_val_ims": 4,
                                    "num_test_ims": 32
                                   },

                file_args = {'labels_file': "/home/evan/data/climate/csv_labels/labels_no_negatives.csv",
                             'tr_data_file': "/home/evan/data/climate/climo_1980.h5",
                              'val_data_file': "/home/evan/data/climate/climo_1981.h5",
                              "test_data_file": "/home/evan/data/climate/climo_1981.h5"},

                data_type = {"data_name": "climate"},
                   


                  
                  )
    


configs = {}
for kwargs in kwargs_dict.values():
    configs.update(kwargs)


