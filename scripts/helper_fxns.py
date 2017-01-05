
import matplotlib; matplotlib.use("agg")


import lasagne
import theano
from theano import tensor as T
import sys
import numpy as np
import json
import pickle
import os
#enable importing of notebooks
# from print_n_plot import plot_ims_with_boxes, add_bbox, plot_im_with_box
from copy import deepcopy
from collections import Counter
from os.path import join, exists
from os import mkdir, makedirs



def makedir_if_not_there(dirname):
    if not exists(dirname):
        try:
            mkdir(dirname)
        except OSError:
            makedirs(dirname)



def softmax3D(x):
    #X is batch_size x n_classes x X x Y tensor
    s = x.shape
    #Flip X to be batch_size*x*y, n_classes -> stack all the one-hot encoded vectors
    x_t = x.transpose((0,2,3,1)).reshape((s[0]*s[2]*s[3],s[1]))
    # take softmax        
    x_sm = T.nnet.softmax(x_t)
    #reshape back to #X is batch_size x n_classes x X x Y tensor
    x_f = x_sm.reshape((s[0],s[2],s[3],s[1])).transpose((0,3,1,2))
    return x_f

def softmax4D(x):
    #X is batch_size x n_classes x time-steps x X x Y tensor
    s = x.shape
    #Flip X to be batch_size*x*y, n_classes -> stack all the one-hot encoded vectors
    x_t = x.transpose((0,2,3,4,1)).reshape((s[0]*s[2]*s[3]*s[4],s[1]))
    # take softmax        
    x_sm = T.nnet.softmax(x_t)
    #reshape back to #X is batch_size x n_classes x X x Y tensor
    x_f = x_sm.reshape((s[0],s[2],s[3],s[4],s[1])).transpose((0,4,1,2,3))
    return x_f



def smoothL1(x):
    #x is vector of scalars
    lto = T.abs_(x)<1
    gteo = T.abs_(x)>=1
    new_x = T.set_subtensor(x[lto.nonzero()],0.5 * T.square(x[lto.nonzero()]))
    new_x = T.set_subtensor(new_x[gteo.nonzero()], T.abs_(new_x[gteo.nonzero()]) - 0.5)
    return new_x



def add_as_appension(key, val, dict_,):
    if key not in dict_:
        dict_[key] = []
    dict_[key].append(val)
    return dict_

def add_as_extension(key, val, dict_):
    if key not in dict_:
        dict_[key] = []
    dict_[key].extend(val)
    return dict_

def add_as_running_total(key,val,dict_):
    if key not in dict_:
        dict_[key] = 0
    dict_[key] += val
    return dict_



class AccuracyGetter(object):
    def __init__(self,kwargs):
        self.kwargs = kwargs
        self.class_name_dict = dict(zip(range(4),["td", "tc", "etc", "ar"], ))

    def get_scores(self, pred_tensor,y_tensor):
        pred_confs_tot = {}
        gt_confs_tot = {"gt_" + str(k):[] for k in range(self.kwargs["num_classes"])}
        pred_confs_tot = {"pred_" + str(k) :[] for k in range(self.kwargs["num_classes"])}
        for i in range(pred_tensor.shape[0]):

            pred_boxes, gt_boxes = self.get_boxes(pred_tensor[i], y_tensor[i])
            pred_confs, gt_confs = self.get_pred_and_gt_scores(pred_boxes,gt_boxes)
            for k in pred_confs.keys():
                pred_confs_tot["pred_" + str(k)].extend(pred_confs[k])
                gt_confs_tot["gt_" + str(k)].extend(gt_confs[k])
                
        pred_confs_tot.update(gt_confs_tot)
        return pred_confs_tot
            


#     def get_ap(self, pred_boxes,gt_boxes):
#         '''pred boxes and gt_boxes is a dictionary key (class integer): value: list of boxes'''
        
#         aps = {k: average_precision_score(gt_confs[k], pred_confs[k])}
    
 
    def get_pred_and_gt_scores(self, pred_boxes,gt_boxes):
        '''input: bbox lists for predicted and ground truth (gt) boxes
           each element in list is a list -> [x, y, w, h, confidence, class]
        output: 
                pred_confs: dictionary, where keys are classes and values are list confidences for each bounding box for that class
                gt_confs: dictionary where keys are classes and values are list of binary variables, corresponding to crrectness of 
                    each element in corresponding lists in pred_confs'''
                
        
        
        #dict where key is class, value is list of boxes predicted as that class
        pred_boxes_by_class = {k: [p for p in pred_boxes if p[5] ==k] for k in range(self.kwargs["num_classes"])}
        
        #dict where key is class, value is list of ground truth boxes for that class
        gt_boxes_by_class = {k:[g for g in gt_boxes if g[5] == k] for k in range(self.kwargs["num_classes"]) }
        
        #dict where key is class, value is list of confidences for each box guess
        pred_confs = {k:[] for k in range(self.kwargs["num_classes"]) }
        
        #dict where key is class, value is list of binary variables, where 1 means, there is a ground truth box
        #for the corresponding guess in pred_confs
        gt_confs = {k:[] for k in range(self.kwargs["num_classes"])}
        

 
        gt_boxes_byc = deepcopy(gt_boxes_by_class)

        #for each predicted box (in order from highest conf to lowest)
        for c in pred_boxes_by_class.keys():
            for pc_box in pred_boxes_by_class[c]:
                conf = pc_box[4]
                cls = pc_box[5]
                ious = np.asarray([self.iou(pc_box, gc_box) for gc_box in gt_boxes_byc[c] ])
                
                # get all gt boxes that have iou with given box over the threshold
                C = [ind for ind, gc_box in enumerate(gt_boxes_byc[c]) if ious[ind] > self.kwargs["iou_thresh"] ]
                
                #this means there is at least once gt box that overlaps over 
                #the IOU threshold with predicted
                if len(C) > 0:
                    # if there are some gt ones that are over the threshold
                    # grab the highest one in terms of iou
                    max_ind = np.argmax(ious)
                    
                    
                    # remove this box from consideration
                    del gt_boxes_byc[c][max_ind]
                    
                    #add the confidence to the prediction
                    pred_confs[c].append(conf)
                    #add a 1 to the ground truth
                    gt_confs[c].append(1)
                
                #no box overlaps with the guess. Put a zero for ground truth here
                else:
                    pred_confs[cls].append(conf)
                    gt_confs[c].append(0)
        

        return pred_confs, gt_confs



    
    def get_all_boxes(self, pred_tensor,y_tensor):
        all_gt_boxes = []
        all_pred_boxes = []
        for i in range(pred_tensor.shape[0]):
            pred_boxes, gt_boxes = self.get_boxes(pred_tensor[i], y_tensor[i])
            all_gt_boxes.append(gt_boxes)
            all_pred_boxes.append(pred_boxes)
        return all_pred_boxes, all_gt_boxes


    def get_boxes(self, pred_tensor, y_tensor):
        ''' pred_tensor is of shape: 5 + num_classes, x_g, y_g 
            y_tensor is same shape'''
        '''returns two dicts of the form
             key (class), value: list of boxes with conf for that class'''
        pred_boxes = self.get_boxes_from_tensor(pred_tensor)
        gt_boxes = self.get_boxes_from_tensor(y_tensor)
        #print len(gt_boxes)
        return pred_boxes, gt_boxes

    def get_boxes_from_tensor(self, tensor):
        #tensor is numpy tensor
        x_g, y_g = tensor.shape[-2], tensor.shape[-1]
        boxes = []

        # pull out all box guesses
        for i in range(x_g):
            for j in range(y_g):
                #print tensor.shape
                coords = tensor[:5,i,j]
                box = self.convert_coords_to_box(coords,i,j)
                conf = coords[-1]
                cls = np.argmax(tensor[6:,i,j]) # + 1 # cuz classes go from 1 to4
                box.append(cls)
                #if conf > self.kwargs["conf_thresh"]:
                boxes.append(box)

        #sort by confidence
        boxes.sort(lambda a,b: -1 if a[4] > b[4] else 1)        
        return boxes


    def convert_coords_to_box(self, coords, xind, yind):
        scale_factor = self.kwargs["scale_factor"]
        #print coords
        xoff,yoff,w,h, conf = coords
        x,y = scale_factor*(xind + xoff), scale_factor *(yind + yoff)

        w,h = 2**w * scale_factor, 2**h * scale_factor

        return [x,y,w,h,conf]

    def iou(self, box1,box2):
        #box1 and box2 are numpy arrays
        #boxes are expected in x_center, y_center, width, height format
        x1,y1,w1,h1 = box1[:4]
        x2,y2,w2,h2 = box2[:4]
        #print w1,h1
        #print w2,h2
        xmin1, xmax1, ymin1, ymax1 = max(0, x1 - w1 / 2.), x1 + w1 /2., max(0,y1 - h1 / 2.), y1 + h1 /2.
        xmin2, xmax2, ymin2, ymax2 = max(0,x2 - w2 / 2.), x2 + w2 /2,max(0,y2 - h2 / 2), y2 + h2 /2
        inters = max(0,(min(xmax1,xmax2) - max(xmin1,xmin2)))   *                               max(0,(min(ymax1,ymax2) - max(ymin1,ymin2)) )
        def get_area(box_mm):
            xmin, xmax, ymin, ymax = box_mm
            area = (xmax - xmin) * (ymax - ymin)
    #         if area == 0.0:
    #             print "aaaaaah", xmin, xmax, ymin, ymax
            return area
        union = get_area((xmin1, xmax1, ymin1, ymax1)) + get_area((xmin2, xmax2, ymin2, ymax2)) - inters                                                        

        return inters / float(union)



def get_detec_loss(pred, gt, kwargs):
    #TODO add in multiple bbox behavior
    #pred is n_ex, [x,y,w,h,c,classes], x, y
    #get number of examples and the indices of the tesnor 
    #to where x,y coirrds height width and confidence go
    pred = pred.transpose((0,2,3,1))
    gt = gt.transpose((0,2,3,1))
    nex = pred.shape[0]
    cinds = T.arange(6)
    
    #x coord indices, y coord indices, width, height, confidence
    xs,ys,ws,hs,cs, csn = cinds[0], cinds[1], cinds[2], cinds[3], cinds[4], cinds[5]
    
    #index for prob vector for all classes
    ps = T.arange(6,pred.shape[3])
    
    #theano will now make elements less than or equal to 0 as zero and others 1 (so output shape is )
    obj_inds = gt[:,:, :,cs] > 0.
   
    #use nonzero in order to get boolean indexing  (eliminate the indices that are zero)
    #get specific x,y location of gt objects and the predicted output for that x,y location
    tg_obj = gt[obj_inds.nonzero()]
    tp_obj = pred[obj_inds.nonzero()]
    
    #term1
    #take the sum of squared difference between predicted and gt for the x and y corrdinate 
    s_x = smoothL1(tp_obj[:,xs] - tg_obj[:,xs])

    s_y = smoothL1(tp_obj[:,ys] - tg_obj[:,ys])

    raw_loss1 = T.sum(s_x + s_y)

    #multipily by lambda coord (the scaling factor for bbox coords)
    sterm1 = kwargs['coord_penalty'] * raw_loss1


    #term2

    #get sum of squared diff of the of heights and widths b/w pred and gt normalized by squared heights and widths of gt 
    s_w = smoothL1(tp_obj[:,ws] - tg_obj[:,ws]) #/ T.square(tg_obj[:,ws])
    s_h = smoothL1((tp_obj[:,hs] - tg_obj[:,hs])) #/ T.square(tg_obj[:,hs])
    raw_loss2 = T.sum(s_w + s_h)

    sterm2 = kwargs['size_penalty'] * raw_loss2


    #term3
    #get sum of squared diff between confidence for places with actual boxes of pred vs. ground truth
    s_c  = -T.log(tp_obj[:,cs]+ 0.00001)
    raw_loss3 = T.sum(s_c)
    sterm3 = raw_loss3


    #term4
    #get the real coordinates where there are no objects
    no_ind  = gt[:,:,:,cs] <= 0.
    tp_no_obj = pred[no_ind.nonzero()]

    #get the log likelhood that there isn't a box
    s_nc = -T.log(tp_no_obj[:,csn]+ 0.00001)

    raw_loss4 = T.sum(s_nc)

    sterm4 = kwargs['nonobj_penalty'] * raw_loss4


    #get the cross entropy of these softmax vectors
    s_p = T.nnet.categorical_crossentropy(tp_obj[:,ps], tg_obj[:,ps])

    raw_loss5 = T.sum(s_p)
    sterm5 = raw_loss5

    #adds up terms divides by number of examples in the batch
    loss = (1. / nex) * (sterm1 + sterm2 + sterm3 + sterm4 + sterm5)
    return loss, [sterm1, sterm2, sterm3, sterm4, sterm5]


# 


def nms(boxes):
    pass



def dump_hyperparams(dic, path):
    new_dic = {k:str(dic[k]) for k in dic.keys()}
    with open(path + '/hyperparams.txt', 'w') as f:
        for k,v in new_dic.iteritems():
            f.write(k + ' : ' + v + "\n")



import logging
def setup_logging(save_path):
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

