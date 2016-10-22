
import matplotlib; matplotlib.use("agg")


import numpy as np
import lasagne
from lasagne.layers import *
import time
import sys
from matplotlib import pyplot as plt
import json
import pickle
from matplotlib import patches
import logging

class TrainVal(object):
    def __init__(self, iterator, tr_kwargs, val_kwargs, num_epochs, fns, save_path, network, n_ims_to_plot=6):
        self.train_errs, self.train_accs, self.val_errs, self.val_accs = [], [], [], []
        self.iterator = iterator
        self.tr_kwargs = tr_kwargs
        self.val_kwargs = val_kwargs
        self.num_epochs = num_epochs
        self.tr_fn, self.val_fn, self.ap_box_fn = fns
        self.n_ims_to_plot = n_ims_to_plot
        self.logger = self.setup_logging(save_path)
        self.epoch = 0
        self.start_time = 0
        self.seed = 5
        self.save_path = save_path
        self.print_network(network)
        self.tr_im =None
        self.tr_boxp = None
        self.val_boxp = None
        self.val_im = None
        
    
    
    def print_network(self,network):
        for layer in get_all_layers(network):
            self.logger.info(str(layer) +' : ' + str(layer.output_shape))
        print count_params(layer)
        
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
            acc, pbox,gbox = self.ap_box_fn(x,y, iou_thresh=0.1, conf_thresh=0.5)
            tr_acc += acc
            tr_batches += 1
        
        self.tr_im = x
        self.tr_boxp = (pbox,gbox)
        assert tr_batches > 0
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
            err = self.val_fn(x,y)
            acc, pbox,gbox = self.ap_box_fn(x,y)
            val_err += err
            val_acc += acc
            val_batches += 1
        
        
        self.val_im = x
        self.val_boxp = (pbox,gbox)
        assert val_batches > 0
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
    
    
    def plot_ims_with_boxes(self):
        tr_pb, tr_gb = self.tr_boxp
        self._plot_im_with_boxes(self.tr_im,tr_pb, tr_gb)
        val_pb, val_gb = self.val_boxp
        self._plot_im_with_boxes(self.val_im,val_pb, val_gb, name="val")
        
        
        
        
    def _plot_im_with_boxes(self,ims, pred_bboxes, gt_bboxes, name="tr", sanity_boxes=None):
        #bbox of form center x,y,w,h
        n_ims = ims.shape[0]
        channels = 1 #ims.shape[1]
        plt.figure(3, figsize=(50,50))
        plt.clf()
        
        
        #sanity boxes is the original bounding boxes
        if sanity_boxes is not None:
            assert np.isclose(gt_bboxes, sanity_boxes).all()

        count=0
        for i in range(n_ims):
            for j in range(channels):  
                count+= 1
                sp = plt.subplot(n_ims,channels, count)
                sp.imshow(ims[i,6])
                classes = [cl for cl in gt_bboxes[i].keys() if len(gt_bboxes[i][cl]) > 0]
                for k in classes:
                    for b,box in enumerate(gt_bboxes[i][k]):
                        self.add_bbox(sp, box, color='g')
                    # top two most confident
                    for pb,pbox in enumerate(pred_bboxes[i][k]):
                        self.add_bbox(sp, pbox, color='r')
                        if pb == 2*b:
                            break
                        
        if self.epoch % 50 == 0:           
            plt.savefig("%s/%s_epoch_%i_boxes.png"%(self.save_path,name,self.epoch))
        plt.savefig("%s/%s_boxes.png"%(self.save_path, name))
        pass


    def add_bbox(self, subplot, bbox, color):
        #box of form center x,y  w,h
        x,y,w,h = bbox[:4]
        subplot.add_patch(patches.Rectangle(
        xy=( y - h / 2., x - w / 2. ,),
        width=h,
        height=w, lw=2,
        fill=False, color=color))


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
        
    

    

def train(iterator, 
          tr_kwargs, 
          val_kwargs,
          network,
          fns, 
          num_epochs, 
          save_weights=False, 
          save_path='./results', 
          load_path=None):
    
    
    print "Starting training..." 
    

    
    tv = TrainVal(iterator, tr_kwargs,val_kwargs, num_epochs, fns, save_path, network)
    for epoch in range(num_epochs):
        tv.train_one_epoch()
        tv.val_one_epoch()
        if epoch % 1 == 0:
            tv.plot_learn_curve()
            tv.plot_ims_with_boxes()
        #if epoch % 100
            
        
        

#             if epoch % 100 == 0 or epoch < 100:
#                 pred_boxes, gt_boxes = box_fn(x_tr,y_tr)              
#                 plot_ims_with_boxes(x_tr[inds], pred_boxes[inds], gt_boxes[inds], epoch=epoch,
#                                     save_plots=save_plots, path=save_path)

            
            
            
            
        


#         if save_weights and epoch % 10 == 0:
  
#             np.savez('%s/model.npz'%(save_path), *lasagne.layers.get_all_param_values(network))





