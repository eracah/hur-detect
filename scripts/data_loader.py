
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



def load_precomputed_data(paths=["/global/project/projectdirs/dasrepo/gordon_bell/climate/data/detection/caffe_data/chur_train.h5",
                                "/global/project/projectdirs/dasrepo/gordon_bell/climate/data/detection/caffe_data/chur_val.h5" ],
                             out_of_core=True):
    #Can't slice the data here b/c slicing an h5py file object will read it into memory and we 
    #want to do out of core
    data = []
    for path in paths:
        h5o = h5py.File(path)
        x,y = h5o['data'], h5o['label']
        if not out_of_core:
            x,y = x[:], y[:]
        data.extend([x,y])
    return data

    
        
    
def load_data(path='/project/projectdirs/dasrepo/gordon_bell/climate/data/detection/hur_train_val.h5',
              num_ims=-1,use_negative=False, scale_factor=16, just_test=False, use_boxes=False):
    
    
    inputs, boxes, labels = load_hurricanes(path, num_ims, use_negative)
    
    xy_dims = np.asarray(inputs.shape[2:])
    d_labels = create_detection_gr_truth(xy_dims, scale_factor, boxes)
    if just_test:
        ret = [inputs, d_labels] if not use_boxes else [inputs, d_labels, boxes]
        return ret
    else:
        x_tr, y_tr, x_val, y_val = set_up_train_test_val(inputs, d_labels)
        return x_tr, y_tr, x_val, y_val


def load_hurricanes(path,num_ims=-1, use_negative=False):

    print 'getting data...'
    h5f = h5py.File(path)
    hshape = h5f['hurs'].shape[0]
    if num_ims == -1:
        excerpt = slice(0,hshape)
    else:
        excerpt = slice(0,num_ims)
    if use_negative:
        excerpt = slice(0,num_ims / 2)
        nhurs = h5f['nhurs'][excerpt]

    hurs = h5f['hurs'][excerpt]
    hur_boxes = h5f['hur_boxes'][excerpt]

    hurs_bboxes = np.asarray(hur_boxes).reshape(hurs.shape[0],4)

    
    #convert from xmin,ymin,xmax,ymax to x_center, y_center, width, heights
    hurs_bboxes = convert_bbox_minmax_to_cent_xywh(hurs_bboxes)

    if use_negative:
        nhurs_bboxes = np.zeros((nhurs.shape[0],4))
        inputs = np.vstack((hurs,nhurs))
        bboxes = np.vstack((hurs_bboxes,nhurs_bboxes))
    else:
        inputs = hurs
        bboxes = hurs_bboxes

    labels = np.zeros((inputs.shape[0]))
    labels[:hurs.shape[0]] = 1
    print inputs.shape[0]
    
    
    return inputs, bboxes, labels





def convert_bbox_minmax_to_cent_xywh(bboxes):
    #current bbox set up is xmin,ymin,xmax,ymax
    xmin, ymin, xmax, ymax = [np.expand_dims(bboxes[:,i], axis=1) for i in range(4)]

    w = xmax - xmin
    h = ymax - ymin
    x_c = xmin + w / 2.
    y_c = ymin + h / 2.
    new_bboxes = np.hstack((x_c, y_c, w, h))
    return new_bboxes
    
    
def get_train_val_test_ix(num_ims):
    # tr, te, val is 0.6,0.2,0.2
    ix = range(num_ims)

    n_val = int(0.2*num_ims)
    n_tr =  num_ims - n_val

    seed = np.random.randint(0,20)
    #shuffle once deterministically
    np.random.RandomState(seed).shuffle(ix)
    val_i = ix[:n_val]
    tr_i = ix[n_val:]

    return tr_i,val_i


def set_up_train_test_val(hurs, labels):

    tr_i, val_i = get_train_val_test_ix(hurs.shape[0])
    


    x_tr, lbl_tr = hurs[tr_i], labels[tr_i]
    
    #normalize data
    x_tr, tr_min, tr_max = normalize(x_tr)
    
    
    # get test and val data and normazlize using statistics from train
    x_val, lbl_val = hurs[val_i], labels[val_i]
    x_val,_,_ = normalize(x_val, tr_min, tr_max)
    
    
    return x_tr, lbl_tr, x_val, lbl_val
    




        
        
        
        
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
    



def create_detection_gr_truth(xy, scale_factor, bbox_array):
    #x_xy : 1,2 tuple with x and y sizes for image
    #scale_factor: factor to scale xy size by fro gr_truth grid for YOLO
    #scale_factor = float(scale_factor)


    #make sure xy coords divide cleanly with scale_factor
    assert not np.any(xy % scale_factor), "scale factor %i must divide the xy (%s) coords cleanly " %(scale_factor, x_xy)
    
    
    x_len,y_len = xy[0] / scale_factor, xy[1] / scale_factor
    last_dim = 6 #x,y,w,h,c plus one binary number for phur or pnot


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
        
        

    return gr_truth

def test_grid(bbox, grid):
    x,y = bbox[0] / 16, bbox[1] / 16
    xo,yo = (bbox[0] % 16) / 16., (bbox[1] % 16) / 16.
    w,h = bbox[2] / 16 /6, bbox[3] / 16/6

    print grid[x,y,:6]
    print np.array([xo,yo,w,h,1.,1.])




if __name__ == "__main__":
    xt,yt,xv,yv= load_data(num_ims=40)





