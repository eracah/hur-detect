
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

from scripts.train_val import train, test, grid_search_val, get_fmaps, get_ims
from scripts.print_n_plot import plot_ims_with_boxes
from scripts.build_network import build_network
from scripts.netcdf_loader import BBoxIterator
from scripts.helper_fxns import setup_logging



default_args = {                  'learning_rate': 0.0001,
                                  'num_tr_days': 365,
                                  'input_shape': (None,16,768,1152),
                                  'dropout_p': 0, 
                                  'weight_decay': 0.0005, 
                                  'num_layers': 6,
                                  'num_extra_conv': 0,
                                  'momentum': 0.9,
                                  'lambda_ae' : 10,
                                  'coord_penalty': 5,
                                  'size_penalty': 7,
                                  'nonobj_penalty': 0.5,
                                  'iou_thresh' : 0.1,
                                  'conf_thresh': 0.8,
                                  'shuffle': False,
                                  "use_fc": False,
                                  'metadata_dir': "/storeSSD/eracah/data/metadata/",
                                  'data_dir': "/storeSSD/eracah/data/netcdf_ims",
                                  'batch_size' : 1,
                                  'epochs': 10000,
                                  'tr_years': [1979,1980,1981,1983,1985,1987],
                                  'val_years': [1982, 1986],
                                  "test_years" : [1984],
                                  'save_weights': True,
                                  'num_classes': 4,
                                  'labels_only': True,
                                  'time_chunks_per_example': 1,
                                  'filter_dim':5,
                                  'scale_factor': 64,
                                  'nonlinearity': LeakyRectify(0.1),
                                  'w_init': HeUniform(),
                                  "batch_norm" : False,
                                  "num_ims_to_plot" : 8,
                                  "test": False,
                                  "get_fmaps": False,
                                  "grid_search": False,
                                  "yolo_batch_norm" : True,
                                  "filters_scale" : 1.,
                                  "yolo_load_path": "None",
                                  "3D": False,
                                  "get_ims": False,
                                  "save_path":"None",
                                  "ignore_plot_fails":1,
                                  "ae_load_path": "None", # "/storeSSD/cbeckham/nersc/models/output/full_image_1/12.model"
                
                                  
                    }



# if inside a notebook, then get rid of weird notebook arguments, so that arg parsing still works
if any(["jupyter" in arg for arg in sys.argv]):
    sys.argv=sys.argv[:1]
    default_args.update({"lambda_ae":0,"yolo_load_path":"./results/run289/models/yolo.pkl", "get_ims":True, "conf_thresh": 0.5 })
    
    

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
for k,v in default_args.iteritems():
    parser.add_argument('--' + k, type=type(v), default=v, help=k)

args = parser.parse_args()



if args.save_path == "None":
    save_path = None
else:
    save_path = args.save_path
run_dir = create_run_dir(save_path)

'''set params'''
kwargs = default_args
kwargs.update(args.__dict__)
if kwargs["lambda_ae"] == 0:
    kwargs["labels_only"] = True
if kwargs["3D"] == True:
    kwargs["labels_only"] = False
    kwargs["input_shape"] = (None,16,8,768,1152)
    kwargs['time_chunks_per_example'] = 8
    
kwargs['num_val_days'] = int(np.ceil(0.2*kwargs['num_tr_days']))
kwargs['num_test_days'] = 365
kwargs['save_path'] = run_dir

'''save hyperparams'''
dump_hyperparams(kwargs,run_dir)

kwargs["logger"] = setup_logging(kwargs['save_path'])

'''get network and train_fns'''
fns, networks = build_network(kwargs)



get_ims(BBoxIterator, kwargs, networks, fns)



if kwargs["test"] == True:
    test(BBoxIterator, kwargs, networks, fns)
elif kwargs["grid_search"] == True:
    grid_search_val(BBoxIterator, kwargs, networks, fns)
elif kwargs["get_fmaps"] == True:
    get_fmaps(BBoxIterator, kwargs, networks, fns)
elif kwargs["get_ims"] == True:
    get_ims(BBoxIterator, kwargs, networks, fns)
else:
    train(BBoxIterator, kwargs, networks, fns)






