
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

from scripts.train_val import *
from scripts.build_network import build_network
from scripts.load_data.netcdf_loader import BBoxIterator
from scripts.helper_fxns import *
from scripts.configs import *



# if inside a notebook, then get rid of weird notebook arguments, so that arg parsing still works
if any(["jupyter" in arg for arg in sys.argv]):
    sys.argv=sys.argv[:1]
    default_args.update({"num_layers": 6, "num_test_days":3,"ignore_plot_fails":0, "test":False, "no_plots":True, "num_filters": 2, "filters_scale": 0.01, "num_tr_days":3, "lambda_ae":0})
    

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
for k,v in default_args.iteritems():
    parser.add_argument('--' + k, type=type(v), default=v, help=k)

args = parser.parse_args()
kwargs = process_kwargs(args.__dict__)



fns, networks = build_network(kwargs)



tv = TrainVal(BBoxIterator,kwargs, fns, networks)
if kwargs["test"] == True:
    tv.test()
else:
    tv.train()


