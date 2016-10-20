
import matplotlib; matplotlib.use("agg")


import sys
import os
import matplotlib
import argparse




'''before we import theano anywhere else we want to make sure we specify 
a unique directory for compiling, so we dont get into a locking issue
if we run multiple hur_mains at once on a global file system. Haven't truly implementedthis yet '''
from scripts.run_dir import create_run_dir
from scripts.helper_fxns import dump_hyperparams
from scripts.data_loader import load_data, load_precomputed_data
#from scripts.train_val import train
from scripts.print_n_plot import plot_ims_with_boxes
from scripts.build_network import build_network
from scripts.netcdf_loader import bbox_iterator



# if inside a notebook, then get rid of weird notebook arguments, so that arg parsing still works
if any(["jupyter" in arg for arg in sys.argv]):
    sys.argv=sys.argv[:1]
    

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=10000,
    help='number of epochs for training')

parser.add_argument('-l', '--learn_rate', default=0.00001, type=float,
    help='the learning rate for the network')

parser.add_argument('-n', '--num_ims', default=6, type=int,
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

parser.add_argument('--num_convpool', default=4, type=int,
    help='number of conv layer-pool layer pairs')

parser.add_argument('--momentum', default=0.9, type=float,
    help='momentum')


args = parser.parse_args()



import numpy as np
import lasagne
import time
import sys
from matplotlib import pyplot as plt
import json
import pickle
from matplotlib import patches
from scripts.helper_fxns import early_stop
import logging



class TrainVal(object):
    def __init__(self, iterator, tr_kwargs, val_kwargs, num_epochs, fns, save_path, n_ims_to_plot=6):
        self.train_errs, self.train_accs, self.val_errs, self.val_accs = [], [], [], []
        self.iterator = iterator
        self.tr_kwargs = tr_kwargs
        self.val_kwargs = val_kwargs
        self.num_epochs = num_epochs
        self.tr_fn, self.val_fn, self.box_fn = fns
        self.n_ims_to_plot = n_ims_to_plot
        self.logger = self.setup_logging(save_path)
        self.epoch = 0
        self.start_time = 0
        self.seed = 5
        self.save_path = save_path
    def train_one_epoch(self):
        self.epoch += 1
        self.start_time = time.time()
        tr_err = 0
        tr_acc = 0
        tr_batches = 0
        start_time = time.time()
        for x,y in self.iterator(**self.tr_kwargs):
            x= np.squeeze(x,axis=2)
            y = np.squeeze(y,axis=1)
            tr_err += self.tr_fn(x,y)
            _, acc = self.val_fn(x,y)
            tr_acc += acc
            tr_batches += 1
        

        self.train_errs.append(tr_err / tr_batches)
        self.train_accs.append(tr_acc / tr_batches)
        self.print_results(tr_err / tr_batches, tr_acc / tr_batches, "train")
    
    def val_one_epoch(self):
        self.start_time = time.time()
        val_err = 0
        val_acc = 0
        val_batches = 0
        for x,y in self.iterator(**self.val_kwargs):
            x= np.squeeze(x,axis=2)
            y = np.squeeze(y,axis=1)
            err, acc = val_fn(x,y)
            val_err += err
            val_acc += acc
            val_batches += 1
        self.val_errs.append(val_err / val_batches)
        self.val_accs.append(val_acc / val_batches)
        self.print_results(val_err / val_batches, val_acc / val_batches,'val')
        

    def print_results(self, err, acc, typ="train"):
        if typ == "train":
            self.logger.info("Epoch {} of {} took {:.3f}s".format(self.epoch, self.num_epochs, time.time() - self.start_time))
        elif typ == "val":
            self.logger.info("\tValidation took {:.3f}s".format(time.time() - self.start_time))
        self.logger.info("\t\t" + typ + " los:\t\t{:.4f}".format(err))
        self.logger.info("\t\t" + typ + "acc:\t\t{:.4f} %".format(acc * 100))
    

    def plot_learn_curve(self):
        self._plot_learn_curve('err')
        self._plot_learn_curve('acc')
        
    def _plot_learn_curve(self,type_):
        plt.figure(1 if type_== 'err' else 2)
        plt.clf()
        plt.title('Train/Val %s' %(type_))
        tr_arr = self.train_errs if type_ == 'err' else self.train_accs
        val_arr = self.val_errs if type_ == 'err' else self.val_accs
        plt.plot(tr_arr, label='train ' + type_)
        plt.plot(val_arr, label='val' + type_)
        plt.legend( loc = 'center left', bbox_to_anchor = (1.0, 0.5),
           ncol=2)

        plt.savefig("%s/%s_learning_curve.png"%(self.save_path,type_))
        pass
    
    
#     def plot_ims_with_boxes(n_ims):
#         for x,y in self.tr_iterator:
#             im = x[0]
            
#     def _plot_im_with_boxes(ims, pred_bboxes, gt_bboxes, sanity_boxes=None):
#         #bbox of form center x,y,w,h
#         n_ims = ims.shape[0]
#         channels = ims.shape[1]
#         plt.figure(1, figsize=(80,80))

#         #sanity boxes is the original bounding boxes
#         if sanity_boxes is not None:
#             assert np.isclose(gt_bboxes, sanity_boxes).all()

#         count=0
#         for i in range(n_ims):
#             for j in range(channels):  
#                 count+= 1
#                 sp = plt.subplot(n_ims,channels, count)
#                 sp.imshow(ims[i,j])
#                 add_bbox(sp, pred_bboxes[i], color='r')
#                 add_bbox(sp, gt_bboxes[i], color='g')
#         if save_plots:
#             plt.savefig("%s/epoch_%i_boxes.png"%(self.save_path,self.epoch))
#             plt.savefig("%s/boxes.png"%(path))
#             pass
#         else:
#             pass


#     def add_bbox(subplot, bbox, color):
#         #box of form center x,y  w,h
#         x,y,w,h = bbox
#         subplot.add_patch(patches.Rectangle(
#         xy=(x - w / 2. , y - h / 2.),
#         width=w,
#         height=h, lw=2,
#         fill=False, color=color))


    def setup_logging(self,save_path):
        logger = logging.getLogger('simple_example')
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler('%s/training.log'%(save_path))
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
        logger.addHandler(fh)
        return logger
        
    

    

def train(iterator, network,
          fns, 
          num_epochs, 
          num_ims=20,
          save_weights=False, 
          save_path='./results', 
          load_path=None):
    
    
    print "Starting training..." 
    
    tr_kwargs = dict(years=[1979], days=num_ims)
    val_kwargs= dict(years=[1980], days=int(0.2*num_ims))
    
    tv = TrainVal(bbox_iterator, tr_kwargs,val_kwargs, num_epochs, fns, save_path)
    for epoch in range(num_epochs):
        tv.train_one_epoch()
        tv.val_one_epoch()
        if epoch % 1 == 0:
            tv.plot_learn_curve()
        #if epoch % 100
            
        
        

#             if epoch % 100 == 0 or epoch < 100:
#                 pred_boxes, gt_boxes = box_fn(x_tr,y_tr)              
#                 plot_ims_with_boxes(x_tr[inds], pred_boxes[inds], gt_boxes[inds], epoch=epoch,
#                                     save_plots=save_plots, path=save_path)

            
            
            
            
        


#         if save_weights and epoch % 10 == 0:
  
#             np.savez('%s/model.npz'%(save_path), *lasagne.layers.get_all_param_values(network))



run_dir = create_run_dir()
print run_dir



'''set params'''
network_kwargs = {'learning_rate': args.learn_rate, 
                  'input_shape': (None,16,768,1152),
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
train_fn, val_fn, box_fn,pred_fn, network, hyperparams = build_network(**network_kwargs)

hyperparams.update({'num_ims': args.num_ims, 'tr_size': args.num_ims})
'''save hyperparams'''
dump_hyperparams(hyperparams, path=run_dir)

'''train'''
train(bbox_iterator, network=network, fns=(train_fn, val_fn, box_fn),num_ims=args.num_ims, save_weights=True, num_epochs=args.epochs, save_path=run_dir)





