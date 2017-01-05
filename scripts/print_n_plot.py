
import matplotlib; matplotlib.use("agg")


import numpy as np
import os
import lasagne
import time
import sys
from matplotlib import pyplot as plt
from matplotlib import patches

from helper_fxns import *

#import data_loader
#from data_loader import load_classification_dataset, load_detection_dataset
#from build_hur_detection_network import build_det_network
#from build_hur_classif_network import build_classif_network



class Plotter(object):
    def __init__(self, kwargs, fns, max_ims, iterator):
        self.kwargs = kwargs
        self.fns = fns
        self.max_ims = max_ims
        self.iterator = iterator
        self.epoch = 0
    def get_encoder_fmaps(self, test_kwargs, tr_kwargs):
        fmaps_tr_dir = join(self.kwargs["save_path"], "tr_fmaps")
        fmaps_te_dir = join(self.kwargs["save_path"], "te_fmaps")
        makedir_if_not_there(fmaps_tr_dir)
        makedir_if_not_there(fmaps_te_dir)
        i =0
        for i, (x,y) in enumerate(self.iterator(**test_kwargs).iterate()):
            fmap = self.fns["hid"](x)
            pickle.dump([x,y,fmap],open(join(fmaps_te_dir, "te_fmaps" + str(i) + ".pkl"), "w"))
            if i + 1 > self.max_ims:
                break


        for i, (x,y) in enumerate(self.iterator(**tr_kwargs).iterate()):
            fmap = self.fns["hid"](x)
            pickle.dump([x,y,fmap],open(join(fmaps_tr_dir, "tr_fmaps" + str(i) + ".pkl"), "w"))
            if i + 1 > self.max_ims:
                break
        
            
        
    def postproc_ims(self, test_kwargs):
        j =0 
        for i, (x,y) in enumerate(self.iterator(**test_kwargs).iterate()):
            if i%4==0:
                j+=1
                self.plot_ims(x,y,"test", j)
        

        

        
        

        
        

    def plot_learn_curve(self, metrics):
        for k in metrics.keys():
            if "time" not in k:
                k = k.replace("tr_", "")
                k = k.replace("val_", "")
                self._plot_learn_curve(k, metrics)
        
    def _plot_learn_curve(self,metric, metrics):
        plt.clf()
        plt.figure(1)
        plt.clf()
        ax = plt.subplot(111)
        ax.set_title('Train/Val %s' %(metric))
        ax.plot(metrics['tr_' + metric], label='train ' + metric)
        ax.plot(metrics['val_' + metric], label='val ' + metric)
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)


        curves_path = join(self.kwargs['save_path'], "learn_curves")
        makedir_if_not_there(curves_path)
        plt.savefig("%s/%s_learning_curve.png"%(curves_path,metric))
        #pass
        plt.clf()
        
    
    def do_plots(self,x,y,type_, num, epoch):
        self.epoch = epoch
        if num >= self.max_ims:
            return
        if self.epoch % 2 == 0:
            if self.kwargs["ignore_plot_fails"]:
                try:
                    self.plot_ims(x, y, type_, num)
                except:
                    pass
            else:
                self.plot_ims(x, y, type_, num)
    
    def plot_ims(self, x, y, type_, num):



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
                    sp.imshow(ims[0,tmq_ind,i][::-1])
                else:
                    sp.imshow(ims[i,tmq_ind][::-1])

                if self.kwargs["3D"]:
                    i = i / 2
                    
                b = 1
                for b,box in enumerate(gt_boxes[i]):
                    self.add_bbox(sp, box, color='g')
                    
                  
                # top two most confident
                ind = 3*len(gt_boxes[i]) if len(pred_boxes[i]) >= 3*b else len(pred_boxes[i])
                if len(gt_boxes[i]) == 0:
                    ind = 3
                for pbox in pred_boxes[i][:ind]:
                    conf = pbox[4]
                    if conf > self.kwargs['conf_thresh']:
                        col = 'r'
                    else:
                        col = 'b'
                    self.add_bbox(sp, pbox, color=col)
         
        
        box_dir = join(self.kwargs['save_path'], name + "_boxes")
        makedir_if_not_there(box_dir)
        

        plt.savefig("%s/%s_epoch_%i_boxes_%i.png"%(box_dir, name, self.epoch, num ))
        self.save_out_boxes(box_dir, "ten_most_conf_boxes_epoch_%i"%(self.epoch), pred_boxes[0], gt_boxes[0])
                
        
        plt.savefig("%s/%s_boxes_%i.png"%(box_dir, name, num))
        self.save_out_boxes(box_dir, "ten_most_conf_boxes", pred_boxes[0], gt_boxes[0])
        
        #pass
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
        x,y,w,h, conf1, cls = bbox
        subplot.add_patch(patches.Rectangle(
        xy=( y - h / 2., (self.xdim - x) - w / 2.),
        width=h,
        height=w, lw=2,
        fill=False, color=color))
        subplot.text(y - h / 2., (self.xdim - x) - w / 2.,self.classes[cls], fontdict={"color":color})

                
#     def plot_reconstructed_ims(self,x,xrec):

            
    def _plot_reconstructed_ims(self, orig, rec, num=1, name="tr"):
        """orig and rec is a N,vars,X,y tensor"""
        """ for 3D they are an N,vars, time_steps,x,y """
        if self.kwargs["3D"]:
            n_ims = orig.shape[2]
        else:
            n_ims = orig.shape[0]
        tmq_ind =6 
        psl_ind = 2
        channels = [tmq_ind, psl_ind] #orig.shape[1] #ims.shape[1]
        plt.figure(4, figsize=(40,40))
        plt.clf()
        

        count=0
        for j in channels:
            for i in range(n_ims):
            
                if self.kwargs["3D"]:
                    im_slice = slice(0,i,j)
                else:
                    im_slice = slice(i,j)
                count += 1
                sp = plt.subplot(n_ims*2,len(channels), count)
                if self.kwargs["3D"]:
                    sp.imshow(orig[0,j, i][::-1])
                else:
                    sp.imshow(orig[i,j][::-1])
                    
                count +=1
                sp = plt.subplot(n_ims*2,len(channels), count)
                if self.kwargs["3D"]:
                    sp.imshow(rec[0,j, i][::-1])
                else:
                    sp.imshow(rec[i,j][::-1])
        
        rec_dir = join(self.kwargs['save_path'], name + "_rec")
        makedir_if_not_there(rec_dir)
        
        
        if self.epoch % 4 == 0:           
            plt.savefig("%s/%s_epoch_%i_rec_%i.png"%(rec_dir, name, self.epoch, num))
        plt.savefig("%s/%s_rec_%i.png"%(rec_dir, name, num))
        #pass
        plt.clf()
        

