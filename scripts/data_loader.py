
import matplotlib; matplotlib.use("agg")


__author__ = 'racah'
import h5py
import numpy as np
from operator import mul
import os
import glob
import sys
import time
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import sys
# from print_n_plot import plot_ims_with_boxes

#1 is hur
#0 is nhur

#TODO try on 96x96 (use bigger file -> get from cori)
def load_hurricane(path,num_ims=-1, detection=False, use_negative=False, seed=7, with_boxes=False):

    print 'getting data...'
    h5f = h5py.File(path)
    if num_ims == -1:
        hurs = h5f['hurs'][:]
        nhurs = h5f['nhurs'][:]
        hur_boxes = h5f['hur_boxes'][:]
    else:
        if use_negative:
            num_each = num_ims / 2
        else:
            num_each = num_ims
        hurs = h5f['hurs'][:num_each]
        nhurs = h5f['nhurs'][:num_each]
        hur_boxes = h5f['hur_boxes'][:num_each]

    hurs_bboxes = np.asarray(hur_boxes).reshape(hurs.shape[0],4)
    nhurs_bboxes = np.zeros((nhurs.shape[0],4))
    
    #convert from xmin,ymin,xmax,ymax to x_center, y_center, width, height
    hurs_bboxes = convert_bbox_minmax_to_cent_xywh(hurs_bboxes)

    if use_negative:
        inputs = np.vstack((hurs,nhurs))
        bboxes = np.vstack((hurs_bboxes,nhurs_bboxes))
    else:
        inputs = hurs
        bboxes = hurs_bboxes


    cl_labels = np.zeros((inputs.shape[0]))
    cl_labels[:hurs.shape[0]] = 1
    if not num_ims:
        num_ims = inputs.shape[0]

    print num_ims


    tr_i, te_i, val_i = get_train_val_test_ix(num_ims, seed)

    return set_up_train_test_val(inputs, bboxes, cl_labels, tr_i, te_i, val_i, detection, with_boxes)


def convert_bbox_minmax_to_cent_xywh(bboxes):
    #current bbox set up is xmin,ymin,xmax,ymax
    xmin, ymin, xmax, ymax = [np.expand_dims(bboxes[:,i], axis=1) for i in range(4)]

    w = xmax - xmin
    h = ymax - ymin
    x_c = xmin + w / 2.
    y_c = ymin + h / 2.
    new_bboxes = np.hstack((x_c, y_c, w, h))
    return new_bboxes
    
    
def get_train_val_test_ix(num_ims,seed):
    # tr, te, val is 0.6,0.2,0.2
    ix = range(num_ims)

    n_te = int(0.2*num_ims)
    n_val = int(0.25*(num_ims - n_te))
    n_tr =  num_ims - n_te - n_val


    #shuffle once deterministically
    np.random.RandomState(seed).shuffle(ix)
    te_i = ix[:n_te]
    rest = ix[n_te:]

    np.random.RandomState(seed * 2).shuffle(rest)
    val_i = rest[:n_val]
    tr_i = rest[n_val:n_val + n_tr]
    return tr_i, te_i, val_i


def set_up_train_test_val(hurs, boxes, cl_labels, tr_i, te_i, val_i, detection, with_boxes):

    #get tr_data
    x_tr, bbox_tr, lbl_tr = hurs[tr_i], boxes[tr_i], cl_labels[tr_i]
    
    #normalize data
    x_tr, tr_min, tr_max = normalize(x_tr)
    
    
    # get test and val data and normazlize using statistics from train
    x_te,bbox_te, lbl_te = hurs[te_i], boxes[te_i], cl_labels[te_i]
    x_te, _ ,_ = normalize(x_te, tr_min, tr_max)
    x_val, bbox_val, lbl_val = hurs[val_i], boxes[val_i], cl_labels[val_i]
    x_val, _ ,_ = normalize(x_val, tr_min, tr_max)
    

    x_dims = hurs.shape[1:]
    
    #get x,y coords of data
    x_xy = np.asarray(x_tr.shape[2:])
    

    
    
    
    y_tr, y_val, y_te = [y.astype('int32') for y in [lbl_tr, lbl_val, lbl_te]]
    if detection:
        grid_tr, grid_te, grid_val = create_detection_gr_truth(x_xy, 16, bbox_tr, bbox_te, bbox_val)
#         test_grid(grid_tr[5],box_tr[5])
        if with_boxes:
            return x_tr, grid_tr, bbox_tr, x_te, grid_te, bbox_te, x_val, grid_val, bbox_val
        else:
            return x_tr, grid_tr, x_te, grid_te, x_val, grid_val
    
    else:
        return x_tr,y_tr, x_te, y_te, x_val,y_val



        
        
        
        
# does standardize and normalize for any axis and will return statistics,
#so you can fit and run on test and validation (both these features do not come together in sklearn)
#otherwise sklearn.preprocessing would be the way to go
def standardize(arr,mean=None,std=None, axis=(0,2,3)):
    if mean is None or std is None:
        mean = arr.mean(axis=axis)
        std = arr.std(axis=axis)
    arr -= mean
    arr /= std
    
    return arr, mean, std
        

def normalize(arr,min_=None, max_=None, axis=(0,2,3)):
        if min_ is None or max_ is None:
            min_ = arr.min(axis=(0,2,3), keepdims=True)

            max_ = arr.max(axis=(0,2,3), keepdims=True)

        midrange = (max_ + min_) / 2.

        range_ = max_ - min_

        arr = (arr - midrange) / (range_ /2.)
        return arr, min_, max_
    

    
#TODO: load a classification dataset and a localization one   
def load_classification_dataset(num_ims=-1, 
                                path='/project/projectdirs/dasrepo/gordon_bell/climate/data/detection/hur_train_val.h5',
                                use_negative=True):
    return load_hurricane(path, num_ims=num_ims, detection=False, use_negative=use_negative)


def load_detection_dataset(num_ims=-1, 
                           path='/project/projectdirs/dasrepo/gordon_bell/climate/data/detection/hur_train_val.h5',
                           use_negative=False, with_boxes=False):
    return load_hurricane(path, num_ims=num_ims, detection=True, use_negative=use_negative, with_boxes=with_boxes)

def create_detection_gr_truth(xy,scale_factor, *bbox_arrays):
    #x_xy : 1,2 tuple with x and y sizes for image
    #scale_factor: factor to scale xy size by fro gr_truth grid for YOLO
    #scale_factor = float(scale_factor)

    gr_truths = []
    #make sure xy coords divide cleanly with scale_factor
    assert not np.any(xy % scale_factor), "scale factor %i must divide the xy (%s) coords cleanly " %(scale_factor, x_xy)
    
    
    x_len,y_len = xy[0] / scale_factor, xy[1] / scale_factor
    last_dim = 6 #x,y,w,h,c plus one binary number for phur or pnot
    
    for bbox_array in bbox_arrays:
        #divide up bbox with has range 0-95 to 0-95/scale_factor (so 6x6 for scale factor of 16)
        bb_scaled = bbox_array / scale_factor
        
        #each coordinate goes at index i,j in the 6x6 array, where i,j are the coordinates of the
        #lower left corner of the grid that center of the box (in 6x6 space ) falls on
        inds = np.floor(bb_scaled[:,:2]).astype('int')
        
        #xywh where x and y are offset from lower left corner of grid thay are in [0,1] and w and h
        # are what fraction the width and height of bboxes are of the total width and total height of the image
        xywh = np.copy(bb_scaled)
        
        #subtract the floored values to get the offset from the grid cell
        xywh[:,:2] -= inds[:,:2].astype('float')
        
        #divide by scaled width and height to get wdith and height relative to width and height of iage
        xywh[:,2] /= x_len
        xywh[:,3] /= y_len
        
        #make gr_truth which is 
        gr_truth = np.zeros((bbox_array.shape[0],x_len ,y_len, last_dim))
        
        #sickens me to a do a for loop here, but numpy ain't cooperating
        # I tried gr_truth[np.arange(gr_truth.shape[0]),inds[:0], inds[:1]][:,4] = xywh
        #but it did not work
        
        # we assume one box per image here
        # for each grid point that is center of image plop in center, and width and height and class
        for i in range(gr_truth.shape[0]):
            #put coordinates
            gr_truth[i,inds[i,0], inds[i,1], :4] = xywh[i]
            
            #put in confidence
            gr_truth[i,inds[i,0], inds[i,1], 4] = 1 if np.sum(xywh[i]) > 0. else 0.
            
            #put in class label
            gr_truth[i,inds[i,0], inds[i,1], 5] = 1 if np.sum(xywh[i]) > 0. else 0.
        
        
        gr_truths.append(gr_truth)
    return gr_truths

def test_grid(bbox, grid):
    x,y = bbox[0] / 16, bbox[1] / 16
    xo,yo = (bbox[0] % 16) / 16., (bbox[1] % 16) / 16.
    w,h = bbox[2] / 16 /6, bbox[3] / 16/6

    print grid[x,y,:6]
    print np.array([xo,yo,w,h,1.,1.])




#test
if __name__ == "__main__":
    x_tr, y_tr,  bbox_tr,grid_tr,    x_te, y_te,bbox_te,grid_te,    x_val,y_val, bbox_val, grid_val = load_detection_dataset(num_ims=40)
    #plot_ims_with_boxes(X_train[:5,0], box_tr, box_tr, epoch=0,save_plots=False, old=False)
#     new_box_tr = convert_bbox_minmax_to_cent_xywh(box_tr)
#     plot_ims_with_boxes(X_train[:5,0], new_box_tr, new_box_tr, epoch=0,save_plots=False, old=False)
    test_grid(bbox_tr[8],grid_tr[8])
        





