
import matplotlib; matplotlib.use("agg")


import sys
import matplotlib
'''enable importing of scripts.'''
import argparse
from scripts.data_loader import load_classification_dataset, load_detection_dataset
from scripts.train_val import train
from scripts.helper_fxns import create_run_dir, dump_hyperparams
from scripts.print_n_plot import plot_ims_with_boxes
from scripts.build_network import build_network



epochs = 1
learning_rate = 0.0001
num_ims = 40




run_dir = create_run_dir()

dataset = load_detection_dataset(num_ims=num_ims)

'''size of ground truth grid'''
grid_size = dataset[1].shape[1]

'''set params'''
network_kwargs = {'learning_rate': learning_rate, 'dropout_p': 0, 'weight_decay': 0, 'num_filters': 1024}
detec_network_kwargs = {'grid_size': grid_size, 'nclass': 1, 'n_boxes':1}


'''get network and train_fns'''
train_fn, val_fn, box_fn, network, hyperparams = build_network(mode='detection', 
                                                               network_kwargs=network_kwargs, 
                                                               detec_specific_kwargs=detec_network_kwargs)

'''save hyperparams'''
dump_hyperparams(hyperparams, path=run_dir)



'''train'''
n = train(dataset, network=network, fns=(train_fn, val_fn, box_fn), num_epochs=epochs, save_path=run_dir)





