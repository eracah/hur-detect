
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
from collections import Counter
from scripts.print_n_plot import *
from scripts.helper_fxns import *
from sklearn.metrics import average_precision_score
import copy

class TrainVal(object):
    def __init__(self, iterator, kwargs, fns, networks):
        self.metrics = {}
        self.kwargs = kwargs
        self.iterator = iterator
        self.fns = fns
        self.xdim=768
        self.ydim = 1152
        self.epoch = 0
        self.start_time = 0
        self.seed = 5
        self.classes = [ "TD", "TC", "ETC", "AR"]
        self.max_ims = self.kwargs['num_ims_to_plot']
        self.networks = networks
        self.print_network(networks)
        it_list= ["variables",'batch_size',"data_dir", "metadata_dir", "shuffle","num_classes",
                    "labels_only","time_chunks_per_example"]
        
        self.std_it_kwargs = {k:kwargs[k] for k in it_list} 
        self.it_kwargs = {k:copy.deepcopy(self.std_it_kwargs) for k in ["tr", "val", "test"]}
        for k,v in self.it_kwargs.iteritems():
            self.it_kwargs[k].update({'days':kwargs['num_'+ k+ '_days'], 'years':kwargs[k +'_years'], "seed": 5})
        
        
    
        
        self.plotter = Plotter(self.kwargs, self.fns, self.max_ims, self.iterator)
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
        self.plotter.plot_learn_curve(self.metrics)
        self.epoch += 1
    def _do_one_epoch(self, type_="tr"):
        print "beginning epoch %i" % (self.epoch)
        start_time = time.time()
        loss_tots = {}
        acc_tots = {}
        batches = 0
        it_kwargs = self.it_kwargs[type_]
        
     
        t0 = time.time()
        for x,y in self.iterator(**it_kwargs).iterate():
            print "total iterate time: ", time.time() -t0
            t= time.time()
            loss_dict, acc_dict = self.do_one_iteration(x,y,type_)
            print "actual iteration time: ", time.time() - t
            for k in loss_dict.keys():
                key = type_ + "_" + k
                loss_tots = add_as_running_total(key,loss_dict[k], loss_tots)
            
                        
            for k in acc_dict.keys():
                key = type_ + "_" + k
                acc_tots = add_as_extension(key,acc_dict[k], acc_tots)
                
            
            if not self.kwargs["no_plots"]:
                self.plotter.do_plots(x, y, type_, batches,self.epoch)
                           
   
            batches += 1
            t0 = time.time()
        t= time.time()
        self.postprocess_epoch(type_, loss_tots,acc_tots, start_time, batches)
        print "postprocess time: ", time.time() - t
      
        if type_ == "val":
            self.save_weights()
            
    def do_one_iteration(self, x,y,type_):
        loss_dict = self.fns[type_](x,y)
            
        acc_dict = self.fns["acc"](x,y, iou_thresh=self.kwargs['iou_thresh'],
                                            conf_thresh=self.kwargs['conf_thresh'])            
           
        return loss_dict, acc_dict #, pred_boxes, gt_boxes
        

    def postprocess_epoch(self, type_, loss_tots,acc_tots, start_time, batches):
        assert batches > 0
                
        loss_tots = {k: v / float(batches) for k,v in loss_tots.iteritems()}
        acc_tots_by_class = {}
        for k in range(self.kwargs["num_classes"]):
            gt_ans = acc_tots[type_ + "_gt_" + str(k)]
            pred_confs = acc_tots[type_ + "_pred_" + str(k)]
            if len(pred_confs) > 0:
                ap = average_precision_score(gt_ans, pred_confs)
                if np.isnan(ap):
                    ap = 0.

            # there are no predictions and no actual labels, so undefined precision
            else:
                ap = np.nan
            acc_tots_by_class[type_ + "_" + self.classes[k] + "_ap"] = ap

        
        #don't penalize for when there are no ground truth of a class
        mAP_list = [a for a in acc_tots_by_class.values() if not np.isnan(a)]
        mAP = np.mean(mAP_list)
        for k,v in loss_tots.iteritems():
            if not isinstance(v, float):
                print "adding ", k, " equal to ", v
            self.metrics = add_as_appension(k,v,self.metrics)
        
        for k,v in acc_tots_by_class.iteritems():
            if not isinstance(v, float):
                print "adding ", k, " equal to ", v
            self.metrics = add_as_appension(k,v,self.metrics)
        
        if not isinstance(mAP, float):
                print "adding ", "map", " equal to ", mAP
        self.metrics = add_as_appension(type_ + "_mAP", mAP,self.metrics)
        
        time_key = type_ + "_time"
        self.metrics = add_as_appension(time_key,time.time() - start_time, self.metrics)
        
        

    
    def save_weights(self):
        #print self.metrics
        max_metrics = ["val_mAP"]
        min_metrics = ["val_loss"]
        for k in max_metrics:
            if len(self.metrics[k]) > 1:
                if self.metrics[k][-1] > max(self.metrics[k][:-1]):
                    self._save_weights("yolo", "best_" + k)
        
        
            else:
                self._save_weights("yolo", "best_" + k)
        for k in min_metrics:
            if len(self.metrics[k]) > 1:
                if self.metrics[k][-1] < min(self.metrics[k][:-1]):
                    self._save_weights("yolo", "best_" + k)





        self._save_weights("yolo", "cur")
        self._save_weights("ae", "cur")
        
    def test(self):
        self._do_one_epoch(type_="test")
        self.print_results(type_="test")
    def val(self):
        self._do_one_epoch(type_="val")
        self.print_results(type_="val")
    
    def train(self):
        for epoch in range(self.kwargs['epochs']):
            self.do_one_epoch()
        
        

        
        
    def _save_weights(self,name,suffix=""):
        params = get_all_param_values(self.networks[name])
        model_dir = join(self.kwargs['save_path'], "models")
        makedir_if_not_there(model_dir)
        pickle.dump(params,open(join(model_dir, name + "_" + suffix + ".pkl"), "w"))
    
    def print_results(self,type_=None):
        self.kwargs['logger'].info("Epoch {} of {}".format(self.epoch + 1, self.kwargs['epochs']))
        for typ in ["tr", "val", "test"]:
            if type_ is not None:
                if typ != type_:
                    continue
            else:
                if typ == "test":
                    continue
            self.kwargs['logger'].info("\t {} took {:.3f}s".format(typ, self.metrics[typ + "_time"][-1]))
            for k,v in self.metrics.iteritems():
                if typ in k[:5] and "time" not in k:
                  
                    if "acc" in k:
                        self.kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f} %".format(v[-1] * 100))
                    else:
                        self.kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f}".format(v[-1]))

        



    
    
# def get_fmaps(iterator, kwargs, networks, fns):
#     tv = TrainVal(iterator,kwargs, fns, networks)
#     tv.get_encoder_fmaps()

# def get_ims(iterator, kwargs, networks, fns):
#     tv = TrainVal(iterator,kwargs, fns, networks)
#     tv.postproc_ims()
    
    
    
         

