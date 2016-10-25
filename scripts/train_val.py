
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
    def __init__(self, iterator, kwargs, fns, networks, n_ims_to_plot=6):
        self.kwargs = kwargs
        self.metrics_keys = ["loss", "yolo_loss", "rec_loss", "acc", "time"]
        self.metrics = {"tr_" + k: [] for k in self.metrics_keys }
        self.metrics.update({"val_" + k: [] for k in self.metrics_keys})
        self.iterator = iterator
        self.fns = fns
       
        self.epoch = 0
        self.start_time = 0
        self.seed = 5

        self.max_ims = n_ims_to_plot
        
        self.print_network(networks)

        it_list= ['batch_size',"data_dir", "metadata_dir", "shuffle","num_classes",
                    "labels_only", "time_chunks_per_example"]
        
        self.tr_kwargs = {k:kwargs[k] for k in it_list} 
        self.tr_kwargs.update({'days':kwargs['num_tr_days'], 'years':kwargs['tr_years']})
        self.val_kwargs = {k:kwargs[k] for k in it_list}
        self.val_kwargs.update({'days':kwargs['num_val_days'], 'years':kwargs['val_years']})
        
    
    def print_network(self, networks):
        yolo, ae = networks['yolo'], networks['ae']
        self._print_network(yolo)
        if self.kwargs['lambda_ae'] != 0:
            
            self._print_network(ae)
            
            
    def _print_network(self, network):
        self.kwargs['logger'].info("\n")
        for layer in get_all_layers(network):
            self.kwargs['logger'].info(str(layer) +' : ' + str(layer.output_shape))
        self.kwargs['logger'].info(str(count_params(layer)))
        self.kwargs['logger'].info("\n")
    
    def do_one_epoch(self):
        self._do_one_epoch(type_="tr")
        self._do_one_epoch(type_="val")
        self.print_results()
        self.epoch += 1
    def _do_one_epoch(self, type_="tr"):
        start_time = time.time()
        metrics_tots = {type_ + "_" + k:0 for k in self.metrics_keys}
        batches = 0
        start_time = time.time()
        if type_ == "tr":
            it_kwargs = self.tr_kwargs
        else:
            it_kwargs = self.val_kwargs
        for x,y in self.iterator(**it_kwargs):
            x= np.squeeze(x,axis=2)
            y = np.squeeze(y,axis=1)
            if type_=="tr":
                loss, yolo_loss, rec_loss = self.fns['tr'](x,y)
            else:
                loss, yolo_loss, rec_loss = self.fns['val'](x,y)
            acc = self.fns["MAP"](x,y, iou_thresh=self.kwargs['iou_thresh'],
                                            conf_thresh=self.kwargs['conf_thresh'])
            
            metrics_tots[type_ + "_rec_loss"] += rec_loss
            metrics_tots[type_ + "_yolo_loss"] += yolo_loss
            metrics_tots[type_ + "_loss"] += loss
            metrics_tots[type_ + "_acc"] += acc
            
   
            batches += 1
        
        assert batches > 0
        for k,v in metrics_tots.iteritems():
            self.metrics[k].append(v / batches)
        self.metrics[type_ + "_time"].append(time.time() - start_time)
    
        

    def print_results(self,):
        self.kwargs['logger'].info("Epoch {} of {} took {:.3f}s".format(self.epoch, self.kwargs['epochs'],
                                                                  self.metrics["tr_time"][-1]))
        for typ in ["tr", "val"]:
            if typ == "val":
                self.kwargs['logger'].info("\tValidation took {:.3f}s".format(self.metrics["val_time"][-1]))
            for k,v in self.metrics.iteritems():
                if typ in k and "time" not in k:
                    if "acc" in k:
                        self.kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f} %".format(v[-1] * 100))
                    else:
                        self.kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f}".format(v[-1]))
        
        

        
 
    

    def plot_learn_curve(self):
        for k in self.metrics_keys:
            self._plot_learn_curve(k)
        
    def _plot_learn_curve(self,type_):
        plt.clf()
        plt.figure(1)
        plt.clf()
        plt.title('Train/Val %s' %(type_))
        plt.plot(self.metrics['tr_' + type_], label='train ' + type_)
        plt.plot(self.metrics['val_' + type_], label='val ' + type_)
        plt.legend( loc = 'center left', bbox_to_anchor = (1.0, 0.5),
           ncol=2)


        plt.savefig("%s/%s_learning_curve.png"%(self.kwargs['save_path'],type_))
        pass
    
    
    def plot_ims_with_boxes(self):
        tr_kwargs = self.tr_kwargs
        tr_kwargs.update({"batch_size":self.max_ims})
        for x,y in self.iterator(**tr_kwargs):
            x= np.squeeze(x,axis=2)
            y = np.squeeze(y,axis=1)
            pred_boxes, gt_boxes = self.fns['box'](x,y)
            self._plot_im_with_boxes(x,pred_boxes, gt_boxes)
            break
         
        val_kwargs = self.val_kwargs
        if val_kwargs['days'] * 4 > self.max_ims:
            val_kwargs.update({"batch_size":self.max_ims})
        else:
            val_kwargs.update({"batch_size":val_kwargs['days'] * 4})
        for x,y in self.iterator(**val_kwargs):
            x= np.squeeze(x,axis=2)
            y = np.squeeze(y,axis=1)
            pred_boxes, gt_boxes = self.fns['box'](x,y)
            self._plot_im_with_boxes(x,pred_boxes, gt_boxes, "val")
            break

        
       
    def _plot_im_with_boxes(self,ims,pred_bboxes, gt_bboxes, name="tr"):
        """ims is a N,vars,X,y tensor
            pred_boxes is a list of dicts where each value of dict is a list of bbox tuples"""
        n_ims = ims.shape[0]
        channels = 1 #ims.shape[1]
        tmq_ind = 6
        plt.figure(3, figsize=(30,30))
        plt.clf()
        

        count=0
        for i in range(n_ims):
            for j in range(channels):
                count+= 1
                sp = plt.subplot(n_ims,channels, count)
                sp.imshow(ims[i,tmq_ind])
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
            plt.savefig("%s/%s_epoch_%i_boxes.png"%(self.kwargs['save_path'],name,self.epoch))
        plt.savefig("%s/%s_boxes.png"%(self.kwargs['save_path'], name))
        pass


    def add_bbox(self, subplot, bbox, color):
        #box of form center x,y  w,h
        x,y,w,h = bbox[:4]
        subplot.add_patch(patches.Rectangle(
        xy=( y - h / 2., x - w / 2. ,),
        width=h,
        height=w, lw=3,
        fill=False, color=color))

                
    def plot_reconstructed_ims(self):
        tr_kwargs = self.tr_kwargs
        tr_kwargs.update({"batch_size": self.max_ims})
        for x,y in self.iterator(**tr_kwargs):
            x= np.squeeze(x,axis=2)
            y = np.squeeze(y,axis=1)
            xrec = self.fns['rec'](x)
            self._plot_reconstructed_ims(x,xrec,name="tr")
            break
         
        val_kwargs = self.val_kwargs
        if val_kwargs['days'] * 4 > self.max_ims:
            val_kwargs.update({"batch_size":self.max_ims})
        else:
            val_kwargs.update({"batch_size":val_kwargs['days'] * 4})
        for x,y in self.iterator(**val_kwargs):
            x= np.squeeze(x,axis=2)
            y = np.squeeze(y,axis=1)
            xrec = self.fns['rec'](x)
            self._plot_reconstructed_ims(x,xrec,name="val")
            break
            
    def _plot_reconstructed_ims(self, orig, rec, name="tr"):
        """orig and rec is a N,vars,X,y tensor"""
        n_ims = orig.shape[0]
        channels = orig.shape[1] #ims.shape[1]
        plt.figure(4, figsize=(30,30))
        plt.clf()
        

        count=0
        for i in range(n_ims):
            for j in range(channels):
                count += 1
                sp = plt.subplot(n_ims*2,channels, count)
                sp.imshow(orig[i,j])
                count +=1
                sp = plt.subplot(n_ims*2,channels, count)
                sp.imshow(rec[i,j])
                        
        if self.epoch % 50 == 0:           
            plt.savefig("%s/%s_epoch_%i_rec.png"%(self.kwargs['save_path'],name,self.epoch))
        plt.savefig("%s/%s_rec.png"%(self.kwargs['save_path'], name))
        pass
        

        
    

    

def train(iterator, kwargs, networks, fns):
    
    
    print "Starting training..." 
    

    
    tv = TrainVal(iterator,kwargs, fns, networks)
    for epoch in range(kwargs['epochs']):
        tv.do_one_epoch()
        if epoch % 1 == 0:
            tv.plot_learn_curve()
            tv.plot_ims_with_boxes()
            if kwargs['lambda_ae'] != 0:
                tv.plot_reconstructed_ims()





