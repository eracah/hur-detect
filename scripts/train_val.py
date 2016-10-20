
import matplotlib; matplotlib.use("agg")


import numpy as np
import lasagne
import time
import sys
from matplotlib import pyplot as plt
import json
import pickle
from matplotlib import patches
from helper_fxns import early_stop
import logging
from data_loader import load_precomputed_data
from build_network import build_network
from run_dir import create_run_dir
from netcdf_loader import bbox_iterator


class TrainVal(object):
    def __init__(self, iterator, tr_kwargs, val_kwargs, num_epochs, fns, save_path, n_ims_to_plot=6):
        self.train_errs, self.train_accs, self.val_errs, self.val_accs = [], [], [], [], []
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
    def train_one_epoch(self, it_kwargs):
        self.epoch += 1
        self.start_time = time.time()
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        for x,y in self.iterator(**self.train_kwargs):
            train_err += self.tr_fn(x,y)
            _, acc = self.val_fn(x,y)
            train_acc += acc
            train_batches += 1
        

        self.train_errs.append(tr_err / tr_batches)
        self.train_accs.append(tr_acc / tr_batches)
        print_results(tr_err / tr_batches, tr_acc / tr_batches, "train")
    
    def val_one_epoch(self, it_kwargs):
        self.start_time = time.time()
        val_err = 0
        val_acc = 0
        val_batches = 0
        for x,y in self.iterator(**self.val_kwargs):
            err, acc = val_fn(x,y)
            val_err += err
            val_acc += acc
            val_batches += 1
        self.val_errs.append(val_err / val_batches)
        self.val_accs.append(val_acc / val_batches)
        print_results(val_err / val_batches, val_acc / val_batches, logger)
        

    def print_results(self, err, acc, typ="train"):
        self.logger.info("Epoch {} of {} took {:.3f}s".format(self.epoch, self.num_epochs, time.time() - self.start_time))
        self.logger.info("\t" + typ + " los:\t\t{:.4f}".format(tr_err))
        self.logger.info("\t" + typ + "acc:\t\t{:.4f} %".format(tr_acc * 100))
    

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

        plt.savefig("%s/%s_learning_curve.png"%(self.save_path))
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
        self.logger = logging.getLogger('simple_example')
        self.logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler('%s/training.log'%(save_path))
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        
    

    

def train(iterator, network,
          fns, 
          num_epochs, 
          num_ims=20,
          save_weights=False, 
          save_path='./results', 
          load_path=None):
    
    
    print "Starting training..." 
    
    tr_kwargs = dict(years=[1979], days=num_ims)
    val_kwargs= dict(years=[1980], days=0.2*num_ims)
    
    tv = TrainVal(bbox_iterator, tr_kwargs,val_kwargs, num_epochs, fns, save_path)
    for epoch in range(num_epochs):
        tv.train_one_epoch()
        tv.val_one_epoch()
        if epoch % 10 == 0:
            tv.plot_learn_curve()
        #if epoch % 100
            
        
        

#             if epoch % 100 == 0 or epoch < 100:
#                 pred_boxes, gt_boxes = box_fn(x_tr,y_tr)              
#                 plot_ims_with_boxes(x_tr[inds], pred_boxes[inds], gt_boxes[inds], epoch=epoch,
#                                     save_plots=save_plots, path=save_path)

            
            
            
            
        


#         if save_weights and epoch % 10 == 0:
  
#             np.savez('%s/model.npz'%(save_path), *lasagne.layers.get_all_param_values(network))




if __name__=="__main__":
    pass





def myit(n=10):
    for i in range(n):
        yield i

