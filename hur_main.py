
import matplotlib; matplotlib.use("agg")


import sys
import matplotlib
import argparse


'''before we import theano anywhere else we want to make sure we specify 
a unique directory for compiling, so we dont get into a locking issue
if we run multiple hur_mains at once on a global file system. Haven't truly implementedthis yet '''
from scripts.run_dir import create_run_dir
from scripts.helper_fxns import dump_hyperparams
from scripts.data_loader import load_classification_dataset, load_detection_dataset
from scripts.train_val import train
from scripts.print_n_plot import plot_ims_with_boxes
from scripts.build_network import build_network



# if inside a notebook, then get rid of weird notebook arguments, so that arg parsing still works
if any(["jupyter" in arg for arg in sys.argv]):
    sys.argv=sys.argv[:1]
    

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=200,
    help='number of epochs for training')

parser.add_argument('-l', '--learn_rate', default=0.0001, type=float,
    help='the learning rate for the network')

parser.add_argument('-n', '--num_ims', default=30, type=int,
    help='number of total images')

parser.add_argument('-f', '--num_filters', default=5, type=int,
    help='number of filters in each conv layer')

parser.add_argument( '--fc', default=10, type=int,
    help='number of fully connected units')

parser.add_argument('--coord_penalty', default=5, type=int,
    help='penalty for guessing coordinates wrong')

parser.add_argument('--size_penalty', default=5, type=int,
    help='penalty for guessing height or width wrong')

parser.add_argument('--nonobj_penalty', default=0.5, type=float,
    help='penalty for guessing an object where one isnt')

parser.add_argument('-c','--num_extra_conv', default=0, type=int,
    help='conv layers to add on to each conv layer before max pooling')

parser.add_argument('--num_convpool', default=4, type=int,
    help='number of conv layer-pool layer pairs')

parser.add_argument('--momentum', default=0.9, type=float,
    help='momentum')


args = parser.parse_args()



run_dir = create_run_dir()
print run_dir
dataset = load_detection_dataset(num_ims=args.num_ims,
                                path='/project/projectdirs/dasrepo/gordon_bell/climate/data/detection/hur_train_val.h5')

'''size of ground truth grid'''
grid_size = dataset[1].shape[1]

'''set params'''
network_kwargs = {'learning_rate': args.learn_rate, 
                  'dropout_p': 0, 
                  'weight_decay': 0, 
                  'num_filters': args.num_filters, 
                  'num_fc_units': args.fc, 
                  'num_convpool': args.num_convpool,
                  'num_extra_conv': args.num_extra_conv,
                  'momentum': args.momentum,
                  'coord_penalty': args.coord_penalty,
                  'nonobj_penalty': args.nonobj_penalty,
                   }


'''get network and train_fns'''
train_fn, val_fn, box_fn, network, hyperparams = build_network(**network_kwargs)

hyperparams.update({'num_ims': args.num_ims, 'tr_size': dataset[0].shape[0]})
'''save hyperparams'''
dump_hyperparams(hyperparams, path=run_dir)

'''train'''
train(dataset, network=network, fns=(train_fn, val_fn, box_fn), save_weights=True, num_epochs=args.epochs, save_path=run_dir)













