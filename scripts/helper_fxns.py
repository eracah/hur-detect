
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



class AccuracyGetter(object):
    def __init__(self,kwargs):
        self.kwargs = kwargs
        self.class_name_dict = dict(zip(range(4),["td", "tc", "etc", "ar"], ))

    def get_MAP(self, pred_tensor,y_tensor):
        # gets two N x 9 x 12 x 18 tensors?
        #return a float, and two lists of N dictionaries
        # each dict has num_class keys pointing to a list of some number of box coords
        #print pred_tensor.shape, y_tensor.shape
    #     print pred_tensor.shape
    #     print y_tensor.shape
#         print sum(
#     (Counter(dict(x)) for x in input),
#     Counter())
        inp = []
        for i in range(pred_tensor.shape[0]):

            pred_boxes, gt_boxes = self.get_boxes(pred_tensor[i], y_tensor[i])
            acc_dict = self.get_ap(pred_boxes, gt_boxes)
            inp.append(acc_dict)
        
        
        sums = sum((Counter(dict(x)) for x in inp),Counter())
        for k in inp[0].keys():
            if k not in sums.keys():
                sums[k] = 0.0
        final_acc_dict = {k: float(v) / pred_tensor.shape[0] for k,v in sums.iteritems() }
#         print "getMap", final_acc_dict
        return final_acc_dict

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






    def get_ap(self, pred_boxes,gt_boxes):
        '''pred boxes and gt_boxes is a dictionary key (class integer): value: list of boxes'''
        chosen_boxes = []
        wrong_chosen_boxes = []
        z = []
        zc = []
        
        #dict matching class to total number of Tp's for that class
        gtd = {k:len([g for g in gt_boxes if g[5] == k]) for k in range(self.kwargs["num_classes"]) }
        pzd = {k:[] for k in range(self.kwargs["num_classes"]) }
        

    # #for each class
    #     #sort boxes by confidence for given class
    #     #print pred_boxes[c]

    #     #print pred_boxes[c]
    #     #grab boxes for that class
    #     pc_boxes = pred_boxes[c]
        gc_boxes = deepcopy(gt_boxes)

        #for each predicted box (in order from highest conf to lowest)
        for pc_box in pred_boxes:
            conf = pc_box[4]
            cls = pc_box[5]
            #print conf
            if conf < self.kwargs["conf_thresh"]:
                continue
            #get all ious for that box with gt box
            ious = np.asarray([self.iou(pc_box, gc_box) for gc_box in gc_boxes ])
            # get all gt boxes that have iou with given box over the threshold
            C = [ind for ind, gc_box in enumerate(gc_boxes) if ious[ind] > self.kwargs["iou_thresh"] ]
            if len(C) > 0:
                # if there are some gt ones that are over the threshold
                # grab the highest one in terms of iou
                max_ind = np.argmax(ious)
                g_cls = gc_box[5]
                # remove this box from consideration
                del gc_boxes[max_ind]
                #plop a true positive into the array

                if cls == g_cls:
                    zc.append(1)
                    pzd[cls].append(1)
                else:
                    zc.append(0)
                    pzd[cls].append(0)
                #keep that box around
                chosen_boxes.append(pc_box)
                z.append(1)
            else:
                pzd[cls].append(0)
                zc.append(0)
                z.append(0)
                # keep that box around also?
                wrong_chosen_boxes.append(pc_box)

        n = len(z)
        tp = sum(z)
        tpc = sum(zc)


        ap = float(tp) / n if n > 0 else 0.
        ar = float(tp) / len(gt_boxes) if len(gt_boxes) > 0 else 0.
        cap = float(tpc) / n if n > 0 else 0.
        car = float(tpc) / len(gt_boxes) if len(gt_boxes) > 0 else 0.
        
        clar = {self.class_name_dict[cls] + "_recall":float(sum(pzd[cls])) / gtd[cls] if gtd[cls] > 0 else 0. for cls in range(self.kwargs["num_classes"]) }
        clap = {self.class_name_dict[cls] + "_precision":float(sum(pzd[cls])) / len(pzd[cls]) if len(pzd[cls]) > 0 else 0. for cls in range(self.kwargs["num_classes"]) }
        d = dict(mAP=ap, mar=ar, mcap=cap, mcar=car, n=n, tp=tp)
        d.update(clar)
        d.update(clap)
#         print "get_ap", d
        return d

            #for pred_clsi_box in pred_clsi_boxes



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
                if conf > self.kwargs["conf_thresh"]:
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
    s_c  = -T.log(tp_obj[:,cs])
    raw_loss3 = T.sum(s_c)
    sterm3 = raw_loss3


    #term4
    #get the real coordinates where there are no objects
    no_ind  = gt[:,:,:,cs] <= 0.
    tp_no_obj = pred[no_ind.nonzero()]

    #get the log likelhood that there isn't a box
    s_nc = -T.log(tp_no_obj[:,csn])

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

