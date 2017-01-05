
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
from helper_fxns import *
from sklearn.metrics import average_precision_score

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
        it_list= ['batch_size',"data_dir", "metadata_dir", "shuffle","num_classes",
                    "labels_only","time_chunks_per_example"]
        
        self.tr_kwargs = {k:kwargs[k] for k in it_list} 
        self.tr_kwargs.update({'days':kwargs['num_tr_days'], 'years':kwargs['tr_years']})
        self.val_kwargs = {k:kwargs[k] for k in it_list}
        self.val_kwargs.update({'days':kwargs['num_val_days'], 'years':kwargs['val_years']})
        self.test_kwargs = {k:kwargs[k] for k in it_list}
        self.test_kwargs.update({'days':kwargs['num_test_days'], 'years':kwargs['test_years'], "seed":5})

    
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
        loss_tots = {}
        acc_tots = {}
        batches = 0
        if type_ == "tr":
            it_kwargs = self.tr_kwargs
        else:
            it_kwargs = self.val_kwargs
            
        for x,y in self.iterator(**it_kwargs).iterate():
            loss_dict, acc_dict = self.do_one_iteration(x,y,type_)
            
            
            for k in loss_dict.keys():
                key = type_ + "_" + k
                loss_tots = add_as_running_total(key,loss_dict[k], loss_tots)
            
                        
            for k in acc_dict.keys():
                key = type_ + "_" + k
                acc_tots = add_as_extension(key,acc_dict[k], acc_tots)
            
            if not self.kwargs["no_plots"]:
                self.do_plots(x, y, type_, batches)
                           
   
            batches += 1

    
        self.postprocess_epoch(type_,loss_tots,acc_tots, start_time, batches)
      
        if type_ == "val":
            self.save_weights()
        

    def postprocess_epoch(self, type_, loss_tots,acc_tots, start_time, batches):
        assert batches > 0
                
        loss_tots = {k: v / float(batches) for k,v in loss_tots.iteritems()}
        acc_tots_by_class = {}
        for k in range(self.kwargs["num_classes"]):
            print self.classes[k]

            acc_tots_by_class[self.classes[k] + "_ap"] = average_precision_score(acc_tots[type_ + "_gt_" + str(k)],
                                                        acc_tots[type_ + "_pred_" + str(k)])
        
        print acc_tots_by_class

        mAP = np.mean(acc_tots_by_class.values())
        
        for k,v in loss_tots.iteritems():
            if not isinstance(v, float):
                print "adding ", k, " equal to ", v
            self.metrics = add_as_appension(type_ + "_" + k,v,self.metrics)
        
        for k,v in acc_tots_by_class.iteritems():
            if not isinstance(v, float):
                print "adding ", k, " equal to ", v
            self.metrics = add_as_appension(type_ + "_" + k,v,self.metrics)
        
        if not isinstance(mAP, float):
                print "adding ", "map", " equal to ", mAP
        self.metrics = add_as_appension(type_ + "_mAP", mAP,self.metrics)
        
        time_key = type_ + "_time"
        self.metrics = add_as_appension(time_key,time.time() - start_time, self.metrics)
        
        
    def do_one_iteration(self, x,y,type_):
        loss_dict = self.fns[type_](x,y)
            
        acc_dict = self.fns["acc"](x,y, iou_thresh=self.kwargs['iou_thresh'],
                                            conf_thresh=self.kwargs['conf_thresh'])            
           
        return loss_dict, acc_dict #, pred_boxes, gt_boxes
    
    def save_weights(self):
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
        
    def _test(self,it_kwargs, iou_thresh=None, conf_thresh=None):
        inps = []
        batches = 0 
        iou_thresh = (iou_thresh if iou_thresh is not None else self.kwargs['iou_thresh'])
        conf_thresh= (conf_thresh if conf_thresh is not None else self.kwargs['conf_thresh'] )
        for x,y in self.iterator(**it_kwargs).iterate():
            acc_dict = self.fns["MAP"](x,y, iou_thresh=iou_thresh,
                                conf_thresh=conf_thresh)
            
            inps.append(acc_dict)
            batches += 1
        
        sums = sum((Counter(dict(x)) for x in inps),Counter())
        for k in inps[0].keys():
            if k not in sums.keys():
                sums[k] = 0.0
        final_acc_dict = {k: v / batches for k,v in sums.iteritems() }
        print final_acc_dict
        del final_acc_dict["n"]
        del final_acc_dict["tp"]
        for k in final_acc_dict.keys():
            self.kwargs["logger"].info("Final " + k +  " for iou: %6.4f and conf: %6.4f is: %6.4f"                                       %(iou_thresh, conf_thresh, final_acc_dict[k]))

        return final_acc_dict
    
    def test(self,iou_thresh=None, conf_thresh=None):
        return self._test(it_kwargs=self.test_kwargs, iou_thresh=iou_thresh, conf_thresh=conf_thresh)
    def val(self,iou_thresh=None, conf_thresh=None):
        return self._test(it_kwargs=self.val_kwargs, iou_thresh=iou_thresh, conf_thresh=conf_thresh)
        
        
    
    def get_encoder_fmaps(self):
        fmaps_tr_dir = join(self.kwargs["save_path"], "tr_fmaps")
        fmaps_te_dir = join(self.kwargs["save_path"], "te_fmaps")
        self.makedir_if_not_there(fmaps_tr_dir)
        self.makedir_if_not_there(fmaps_te_dir)
        i =0
        for i, (x,y) in enumerate(self.iterator(**self.test_kwargs).iterate()):
            fmap = self.fns["hid"](x)
            pickle.dump([x,y,fmap],open(join(fmaps_te_dir, "te_fmaps" + str(i) + ".pkl"), "w"))
            if i + 1 > self.max_ims:
                break


        for i, (x,y) in enumerate(self.iterator(**self.tr_kwargs).iterate()):
            fmap = self.fns["hid"](x)
            pickle.dump([x,y,fmap],open(join(fmaps_tr_dir, "tr_fmaps" + str(i) + ".pkl"), "w"))
            if i + 1 > self.max_ims:
                break
        
            
        
    def postproc_ims(self):
        j =0 
        for i, (x,y) in enumerate(self.iterator(**self.test_kwargs).iterate()):
            if i%4==0:
                j+=1
                self.plot_ims(x,y,"test", j)
        
    def _save_weights(self,name,suffix=""):
        params = get_all_param_values(self.networks[name])
        model_dir = join(self.kwargs['save_path'], "models")
        self.makedir_if_not_there(model_dir)
        pickle.dump(params,open(join(model_dir, name + "_" + suffix + ".pkl"), "w"))
        
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
                    print v[-1]
                    if "acc" in k:
                        self.kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f} %".format(v[-1] * 100))
                    else:
                        self.kwargs['logger'].info("\t\t" + k + ":\t\t{:.4f}".format(v[-1]))
        
        

    def plot_learn_curve(self):
        for k in self.metrics.keys():
            if "time" not in k:
                k = k.replace("tr_", "")
                k = k.replace("val_", "")
                self._plot_learn_curve(k)
        
    def _plot_learn_curve(self,metric):
        plt.clf()
        plt.figure(1)
        plt.clf()
        ax = plt.subplot(111)
        ax.set_title('Train/Val %s' %(metric))
        ax.plot(self.metrics['tr_' + metric], label='train ' + metric)
        ax.plot(self.metrics['val_' + metric], label='val ' + metric)
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)


        curves_path = join(self.kwargs['save_path'], "learn_curves")
        self.makedir_if_not_there(curves_path)
        plt.savefig("%s/%s_learning_curve.png"%(curves_path,metric))
        #pass
        plt.clf()
        
    
    def do_plots(self,x,y,type_, num):
        if num >= self.max_ims:
            return
        if self.epoch % 2 == 0:
            if self.kwargs["ignore_plot_fails"]:
                try:
                    self.plot_ims(x, y, type_, batches)
                except:
                    pass
            else:
                self.plot_ims(x, y, type_, batches)
    
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
        self.makedir_if_not_there(box_dir)
        

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
        self.makedir_if_not_there(rec_dir)
        
        
        if self.epoch % 4 == 0:           
            plt.savefig("%s/%s_epoch_%i_rec_%i.png"%(rec_dir, name, self.epoch, num))
        plt.savefig("%s/%s_rec_%i.png"%(rec_dir, name, num))
        #pass
        plt.clf()
        

        

def train(iterator, kwargs, networks, fns):
    print "Starting training..." 
    

    
    tv = TrainVal(iterator,kwargs, fns, networks)
    for epoch in range(kwargs['epochs']):
        tv.do_one_epoch()
        tv.plot_learn_curve()
        
def grid_search_val(iterator, kwargs, networks, fns):
    tv = TrainVal(iterator,kwargs, fns, networks)
    max_ = (0.0,0.0,0.0, 0.0,0,0)
    
    iou_params = [0.1]
    conf_params = [0.4, 0.6, 0.8, 0.9]
    for iou_thresh in iou_params:
        for conf_thresh in conf_params:
            final_acc_dict = tv.val(iou_thresh=iou_thresh, conf_thresh=conf_thresh)
            if final_acc_dict["mAP"] >=  max_[0]:
                max_ = [final_acc_dict[k] for k in ["mAP", "mar", "mcap", "mcar"]] + [iou_thresh, conf_thresh]
                kwargs["logger"].info(str(max_))
    amax = zip(["mAP", "mar", "mcap", "mcar"] + ["iou_thresh", "conf_thresh"], max_)
    kwargs["logger"].info("Final Params: " + str(amax))
    
def test(iterator, kwargs, networks, fns):
    tv = TrainVal(iterator,kwargs, fns, networks)
    final_acc_dict = tv.test()
    

def get_fmaps(iterator, kwargs, networks, fns):
    tv = TrainVal(iterator,kwargs, fns, networks)
    tv.get_encoder_fmaps()

def get_ims(iterator, kwargs, networks, fns):
    tv = TrainVal(iterator,kwargs, fns, networks)
    tv.postproc_ims()
    
    
    
         



a="tr_bool"



a.replace("mon", "")





