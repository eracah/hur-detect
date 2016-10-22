
import matplotlib; matplotlib.use("agg")


import sys
import os
import matplotlib
import argparse
import numpy as np



'''before we import theano anywhere else we want to make sure we specify 
a unique directory for compiling, so we dont get into a locking issue
if we run multiple hur_mains at once on a global file system. Haven't truly implementedthis yet '''
from scripts.run_dir import create_run_dir
from scripts.helper_fxns import dump_hyperparams

from scripts.train_val import train
from scripts.print_n_plot import plot_ims_with_boxes
from scripts.build_network import build_network
from scripts.netcdf_loader import bbox_iterator



# if inside a notebook, then get rid of weird notebook arguments, so that arg parsing still works
if any(["jupyter" in arg for arg in sys.argv]):
    sys.argv=sys.argv[:1]
    

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=10000,
    help='number of epochs for training')

parser.add_argument('-d', '--data_path', type=str, default="/storeSSD/eracah/data/netcdf_ims",
    help='path to data')

parser.add_argument('-m', '--metadata_path', type=str, default="/storeSSD/eracah/data/metadata/",
    help='path to data')

parser.add_argument('-l', '--learn_rate', default=0.00001, type=float,
    help='the learning rate for the network')

parser.add_argument('-n', '--num_ims', default=2, type=int,
    help='number of total images')

parser.add_argument('-f', '--num_filters', default=2, type=int,
    help='number of filters in each conv layer')

parser.add_argument( '--fc', default=512, type=int,
    help='number of fully connected units')

parser.add_argument('--coord_penalty', default=5, type=int,
    help='penalty for guessing coordinates wrong')

parser.add_argument('--size_penalty', default=5, type=int,
    help='penalty for guessing height or width wrong')

parser.add_argument('--nonobj_penalty', default=0.5, type=float,
    help='penalty for guessing an object where one isnt')

parser.add_argument('-c','--num_extra_conv', default=0, type=int,
    help='conv layers to add on to each conv layer before max pooling')
parser.add_argument('-b','--batch_size', default=1, type=int,
    help='batch size')

parser.add_argument('--num_convpool', default=4, type=int,
    help='number of conv layer-pool layer pairs')

parser.add_argument('--momentum', default=0.9, type=float,
    help='momentum')


args = parser.parse_args()



run_dir = create_run_dir()
print run_dir



'''set params'''
network_kwargs = {'learning_rate': args.learn_rate, 
                  'input_shape': (None,16,768,1152),
                  'dropout_p': 0, 
                  'weight_decay': 0.0005, 
                  'num_filters': args.num_filters, 
                  'num_fc_units': args.fc, 
                  'num_convpool': args.num_convpool,
                  'num_extra_conv': args.num_extra_conv,
                  'momentum': args.momentum,
                  'coord_penalty': args.coord_penalty,
                  'nonobj_penalty': args.nonobj_penalty,
                   }


dir_kwargs = dict(data_dir=args.data_path, metadata_dir=args.metadata_path, shuffle=True, batch_size=args.batch_size)
tr_kwargs = dict(years=[1979, 1980,1981], days=args.num_ims)
tr_kwargs.update(dir_kwargs)
val_kwargs= dict(years=[1982,1983,1984,1985,1986], days=int(np.ceil(0.2*args.num_ims)))
val_kwargs.update(dir_kwargs)

'''get network and train_fns'''
train_fn, val_fn,pred_fn, ap_box_fn, network, hyperparams = build_network(**network_kwargs)

hyperparams.update({'num_ims': args.num_ims, 'tr_size': args.num_ims})
'''save hyperparams'''
dump_hyperparams(hyperparams, path=run_dir)



'''train'''
train(bbox_iterator,tr_kwargs,val_kwargs, network=network, fns=(train_fn, val_fn, ap_box_fn), save_weights=True, num_epochs=args.epochs, save_path=run_dir)





