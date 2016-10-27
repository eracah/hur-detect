
import matplotlib; matplotlib.use("agg")


import sys
import os
import matplotlib
import argparse
import numpy as np
from lasagne.nonlinearities import *
from lasagne.init import *

'''before we import theano anywhere else we want to make sure we specify 
a unique directory for compiling, so we dont get into a locking issue
if we run multiple hur_mains at once on a global file system. Haven't truly implementedthis yet '''
from scripts.run_dir import create_run_dir
from scripts.helper_fxns import dump_hyperparams

from scripts.train_val import train, test
from scripts.print_n_plot import plot_ims_with_boxes
from scripts.build_network import build_network
from scripts.netcdf_loader import bbox_iterator
from scripts.helper_fxns import setup_logging



default_args = {                  'learning_rate': 0.0001,
                                  'num_tr_days': 150,
                                  'input_shape': (None,16,768,1152),
                                  'dropout_p': 0, 
                                  'weight_decay': 0.0005, 
                                  'num_filters': 128, 
                                  'num_layers': 6,
                                  'num_extra_conv': 0,
                                  'momentum': 0.9,
                                  'lambda_ae' : 10,
                                  'coord_penalty': 5,
                                  'size_penalty': 5,
                                  'nonobj_penalty': 0.5,
                                  'iou_thresh' : 0.1,
                                  'conf_thresh': 0.4,
                                  'shuffle': True,
                                  'num_fc_units': "None",
                                  'metadata_dir': "/storeSSD/eracah/data/metadata/",
                                  'data_dir': "/storeSSD/eracah/data/netcdf_ims",
                                  'batch_size' : 1,
                                  'ae_weight': 0.0,
                                  'epochs': 10000,
                                  'tr_years': [1980,1983],
                                  'val_years': [1982],
                                  'save_weights': True,
                                  'num_classes': 4,
                                  'labels_only': True,
                                  'time_chunks_per_example': 1,
                                  'filter_dim':5,
                                  'scale_factor': 64,
                                  'nonlinearity': LeakyRectify(0.1),
                                  'w_init': HeUniform(),
                                  "batch_norm" : False,
                                  "num_ims_to_plot" : 3,
                                  "test": False,
                                  "yolo_batch_norm" : True,
                                  "yolo_load_path": "None",
                                  "ae_load_path": "None", # "/storeSSD/cbeckham/nersc/models/output/full_image_1/12.model"
                
                                  
                    }



# if inside a notebook, then get rid of weird notebook arguments, so that arg parsing still works
if any(["jupyter" in arg for arg in sys.argv]):
    sys.argv=sys.argv[:1]
    default_args.update({"num_tr_days":2, "num_filters":2, "num_layers": 1})
    
    

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
for k,v in default_args.iteritems():
    parser.add_argument('--' + k, type=type(v), default=v, help=k)

args = parser.parse_args()



run_dir = create_run_dir()

'''set params'''
kwargs = default_args
kwargs.update(args.__dict__)
kwargs['num_val_days'] = int(np.ceil(0.2*kwargs['num_tr_days']))
kwargs['save_path'] = run_dir

'''save hyperparams'''
dump_hyperparams(kwargs,run_dir)

kwargs["logger"] = setup_logging(kwargs['save_path'])



'''get network and train_fns'''
fns, networks = build_network(kwargs)



if kwargs["test"] == True:
    test(bbox_iterator, kwargs, networks, fns)
else:
    train(bbox_iterator, kwargs, networks, fns)






