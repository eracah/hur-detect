
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
from os.path import join, exists
from os import mkdir, makedirs



class TrainVal(object):
    def __init__(self, iterator, kwargs, fns, networks):
        self.kwargs = kwargs
        self.metrics_keys = ["loss", "yolo_loss", "rec_loss", "acc", "time", "raw_yolo_loss",                              "weight_decay_term", "true_positives", "total_guesses", "mean_average_recall"] +                             ['coord_term', 'size_term', 'conf_term', 'no_obj_conf_term', 'xentropy_term']
        self.metrics = {"tr_" + k: [] for k in self.metrics_keys }
        self.metrics.update({"val_" + k: [] for k in self.metrics_keys})
        self.iterator = iterator
        self.fns = fns
       
        self.epoch = 0
        self.start_time = 0
        self.seed = 5

        self.max_ims = self.kwargs['num_ims_to_plot']
        self.networks = networks
        self.print_network(networks)
        it_list= ['batch_size',"data_dir", "metadata_dir", "shuffle","num_classes",
                    "labels_only","time_chunks_per_example"]
        
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
        print "beginning epoch %i" % (self.epoch)
        start_time = time.time()
        metrics_tots = {type_ + "_" + k:0 for k in self.metrics_keys}
        batches = 0
        if type_ == "tr":
            it_kwargs = self.tr_kwargs
        else:
            it_kwargs = self.val_kwargs
        for x,y in self.iterator(**it_kwargs).iterate():
            
           
            
            if self.kwargs['lambda_ae'] != 0:
                loss, yolo_loss, rec_loss = self.fns[type_](x,y)
            else:
                w_yolo_loss, raw_yolo_loss, weight_decay_term, coord_term, size_term, conf_term, no_obj_conf_term, xentropy_term = self.fns[type_](x,y)
                yolo_loss = w_yolo_loss
                loss = yolo_loss
                terms = [coord_term, size_term, conf_term, no_obj_conf_term, xentropy_term]
                rec_loss = np.inf
            
            acc, tp, n, MAR = self.fns["MAP"](x,y, iou_thresh=self.kwargs['iou_thresh'],
                                            conf_thresh=self.kwargs['conf_thresh'])
            
            pred_boxes, gt_boxes = self.fns["box"](x,y)
        
            
            
            
            if self.epoch % 2 == 0:
                self.plot_ims(x, y, type_, batches)
            
    
            
            for t, tk in enumerate(['coord_term', 'size_term', 'conf_term', 'no_obj_conf_term', 'xentropy_term']):
                metrics_tots[type_ + "_" + tk] += terms[t]
                
            metrics_tots[type_ + "_weight_decay_term"] += weight_decay_term
            metrics_tots[type_ + "_raw_yolo_loss"] += raw_yolo_loss
            metrics_tots[type_ + "_rec_loss"] += rec_loss
            metrics_tots[type_ + "_yolo_loss"] += yolo_loss
            metrics_tots[type_ + "_loss"] += loss
            metrics_tots[type_ + "_acc"] += acc
            metrics_tots[type_ + "_true_positives"] += tp
            metrics_tots[type_ + "_total_guesses"] += n
            metrics_tots[type_ + "_mean_average_recall"] += MAR
            
   
            batches += 1
        del x
        del y
        assert batches > 0
        for k,v in metrics_tots.iteritems():
            self.metrics[k].append(v / batches)
        self.metrics[type_ + "_time"].append(time.time() - start_time)
        self.save_weights("yolo")
        self.save_weights("ae")
        
    
    def test(self,iou_thresh=None, conf_thresh=None):
        it_kwargs = self.val_kwargs #test_kwargs
        acc_tot = 0
        batches = 0
        iou_thresh = (iou_thresh if iou_thresh is not None else self.kwargs['iou_thresh'])
        conf_thresh= (conf_thresh if iou_thresh is not None else self.kwargs['conf_thresh'] )
        for x,y in self.iterator(**it_kwargs).iterate():
            acc, tp, n, MAR = self.fns["MAP"](x,y, iou_thresh=iou_thresh,
                                conf_thresh=conf_thresh)
            acc_tot += acc
            batches += 1
        
        MAP = float(acc_tot) / batches
        self.kwargs["logger"].info("Final Mean Average Precision is: %6.4f" % MAP)
        return MAP
            

    def save_weights(self,name):
        params = get_all_param_values(self.networks[name])
        model_dir = join(self.kwargs['save_path'], "models")
        self.makedir_if_not_there(model_dir)
        pickle.dump(params,open(join(model_dir, name + ".pkl"), "w"))
        
    def makedir_if_not_there(self, dirname):
        if not exists(dirname):
            try:
                mkdir(dirname)
            except OSError:
                makedirs(dirname)
        
        
    def print_results(self):
        self.kwargs['logger'].info("Epoch {} of {} took {:.3f}s".format(self.epoch + 1, self.kwargs['epochs'],
                                                                  self.metrics["tr_time"][-1]))
        for typ in ["tr", "val"]:
            if typ == "val":
                self.kwargs['logger'].info("\tValidation took {:.3f}s".format(self.metrics["val_time"][-1]))
            for k,v in self.metrics.iteritems():
                if typ in k[:4] and "time" not in k:
                    if "acc" in k:
                        self.kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f} %".format(v[-1] * 100))
                    else:
                        self.kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f}".format(v[-1]))
        
        

    def plot_learn_curve(self):
        for k in self.metrics_keys:
            if "time" not in k:
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

        curves_path = join(self.kwargs['save_path'], "learn_curves")
        self.makedir_if_not_there(curves_path)
        plt.savefig("%s/%s_learning_curve.png"%(curves_path,type_))
        pass
        plt.clf()
        
    
    
    def plot_ims(self, x, y, type_, num):
        if num >= self.max_ims:
            return
        pred_boxes, gt_boxes = self.fns['box'](x,y)
        self._plot_im_with_boxes(x,pred_boxes, gt_boxes, num=num, name=type_)
        if self.kwargs['lambda_ae'] != 0:
            xrec = self.fns['rec'](x)
            self._plot_reconstructed_ims(x,xrec,num=num, name=type_)


        
       
    def _plot_im_with_boxes(self,ims,pred_boxes, gt_boxes, num=1, name="tr"):
        """ims is a N,vars,X,y tensor
            pred_boxes is a list of dicts where each value of dict is a list of bbox tuples"""
        if self.kwargs["3D"]:
            n_ims = ims.shape[2]
        else:
            n_ims = ims.shape[0]

#         print len(pred_boxes)
#         print len(gt_boxes)
#         print len(pred_boxes[0])
#         print len(gt_boxes[0])
        channels = 1 #ims.shape[1]
        tmq_ind = 6
        if self.kwargs["3D"]:
            plt.figure(3, figsize=(50,50))
        else:
            plt.figure(3, figsize=(7,7))
        plt.clf()
        

        count=0
        for i in range(n_ims):
            if self.kwargs["3D"]:
                if i %2 != 0:
                    continue
            for j in range(channels):
                count+= 1
                sp = plt.subplot(n_ims,channels, count)
                if self.kwargs["3D"]:
                    sp.imshow(ims[0,tmq_ind,i])
                else:
                    sp.imshow(ims[i,tmq_ind])

                if self.kwargs["3D"]:
                    i = i / 2
                for b,box in enumerate(gt_boxes[i]):
                    self.add_bbox(sp, box, color='g')
                    
                    
                # top two most confident
                ind = 3*b if len(pred_boxes[i]) >= 3*b else len(pred_boxes[i])
                for pbox in pred_boxes[i][:ind]:
                    conf = pbox[4]
                    if conf > self.kwargs['conf_thresh']:
                        col = 'r'
                    else:
                        col = 'b'
                    self.add_bbox(sp, pbox, color=col)
         
        
        box_dir = join(self.kwargs['save_path'], name + "_boxes")
        self.makedir_if_not_there(box_dir)
        

        plt.savefig("%s/%s_epoch_%i_boxes_%i.png"%(box_dir, name, self.epoch, num ))
        self.save_out_boxes(box_dir, "ten_most_conf_boxes_epoch_%i"%(self.epoch), pred_boxes[0], gt_boxes[0])
                
        
        plt.savefig("%s/%s_boxes_%i.png"%(box_dir, name, num))
        self.save_out_boxes(box_dir, "ten_most_conf_boxes", pred_boxes[0], gt_boxes[0])
        
        pass
        plt.clf()

    
    def save_out_boxes(self,box_dir, name, pred_boxes, gt_boxes):
        pred_boxes.sort(lambda a,b: -1 if a[4] > b[4] else 1) 
        with open(join(box_dir, name), "a") as f:
            ind = 10 if len(pred_boxes) >= 10 else len(pred_boxes)
            for box in pred_boxes[:ind]:
                f.write("\n" + str(box) + "\n")
            ind = ind = 10 if len(gt_boxes) >= 10 else len(gt_boxes)
            f.write("\n ooooh gt boxes: \n")
            for box in gt_boxes[:ind]:
                f.write(str(box) + "\n")
            f.write("\n\n\n")
            

    def add_bbox(self, subplot, bbox, color):
        #box of form center x,y  w,h
        x,y,w,h = bbox[:4]
        subplot.add_patch(patches.Rectangle(
        xy=( y - h / 2., x - w / 2. ,),
        width=h,
        height=w, lw=1,
        fill=False, color=color))

                
#     def plot_reconstructed_ims(self,x,xrec):

            
    def _plot_reconstructed_ims(self, orig, rec, num=1, name="tr"):
        """orig and rec is a N,vars,X,y tensor"""
        n_ims = orig.shape[0]
        channels = orig.shape[1] #ims.shape[1]
        plt.figure(4, figsize=(40,40))
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
        
        rec_dir = join(self.kwargs['save_path'], name + "_rec")
        self.makedir_if_not_there(rec_dir)
        
        
        if self.epoch % 10 == 0:           
            plt.savefig("%s/%s_epoch_%i_rec_%i.png"%(rec_dir, name, self.epoch, num))
        plt.savefig("%s/%s_rec_%i.png"%(rec_dir, name, num))
        pass
        plt.clf()
        

        
    

    

def train(iterator, kwargs, networks, fns):
    
    
    print "Starting training..." 
    

    
    tv = TrainVal(iterator,kwargs, fns, networks)
    for epoch in range(kwargs['epochs']):
        tv.do_one_epoch()
        tv.plot_learn_curve()
        
def test(iterator, kwargs, networks, fns):
    tv = TrainVal(iterator,kwargs, fns, networks)
    max_ = (0.0,0,0)
    
    iou_params = [0.1]
    conf_params = [0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8,0.9]
    for iou_thresh in iou_params:
        for conf_thresh in conf_params:
            MAP = tv.test(iou_thresh=iou_thresh, conf_thresh=conf_thresh)
            if MAP >=  max_[0]:
                print MAP, iou_thresh, conf_thresh
                max_ = (MAP, iou_thresh, conf_thresh)
    print max_
         





