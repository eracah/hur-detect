


import re
import numpy as np
import datetime as dt
import os
import sys
import numpy as np
import time
import h5py
import os
from os.path import join



def correct_class_labels(lbls, tf_mode=True):
    """subtract class labels by 1, so labels used to be 1-4 now 0-3 and 0 is still 0"""
    if tf_mode:
        cl_index = 5
        lbls[:,:,:,cl_index] = lbls[:,:,:,cl_index] - 1
        lbls[:,:,:,cl_index] = np.where(lbls[:,:,:,cl_index]==-1,
                                        np.zeros_like(lbls[:,:,:,cl_index]),
                                        lbls[:,:,:,cl_index] )
    else:
        assert False, "not implemented"
    return lbls



def convert_bbox_minmax_to_cent_xywh(bboxes):
    #current bbox set up is xmin,ymin,xmax,ymax
    xmin, xmax,ymin,  ymax = [ bboxes[:,i] for i in range(4) ]
    
    w = xmax - xmin
    h = ymax - ymin

    x_c = xmin + w / 2.
    y_c = ymin + h / 2.
    
    
    bboxes[:,0] = x_c
    bboxes[:,1] = y_c
    bboxes[:,2] = w # w
    bboxes[:,3] = h #h
    return bboxes



def get_timestamp(filename):
    #print filename
    rpyear = re.compile(r"(\.h2\.)(.*?)(-)")
    rpdaymonth = re.compile(r"(-)(.*?)(\d{5}\.)")
    year=int(rpyear.search(filename).groups()[1])
    tmp=rpdaymonth.search(filename).groups()[1].split('-')
    month=int(tmp[0])
    day=int(tmp[1])
    return dt.date(year,month,day)



def get_xy_inds(x,y, scale_factor):
        # get the indices to the lower left corner of the grid
        
        #scale x and y down
        xs, ys = x / scale_factor, y / scale_factor
        eps = 10*np.finfo(float).eps
        #take the floor of x and y -> which is rounding to nearest bottom left corner
        x_ind, y_ind = [np.floor(k - 10*eps ).astype('int') for k in [xs,ys]]
        return x_ind, y_ind
    

def get_xy_offsets(x,y, x_ind, y_ind, scale_factor):
    #scale x and y down
    xs, ys = x / scale_factor, y / scale_factor
    
    #get the offsets by subtracting the scaled lower left corner coords from the scaled real coords
    xoff, yoff = xs - x_ind, ys - y_ind
    
    return xoff, yoff

def get_parametrized_wh(w,h,scale_factor):
    scale_factor = float(scale_factor)
    ws , hs = w / scale_factor, h/ scale_factor
    assert ws !=0 and hs!=0, "ahhhhh"
    wp, hp = np.log2(ws), np.log2(hs)
    return wp, hp
    

def convert_class_to_one_hot(class_num, num_classes):
    vec = num_classes * [0]
    vec[class_num - 1] = 1
    return vec

def get_box_vector(coords, scale_factor, num_classes, caffe_format):
    x,y,w,h,cls = coords
    xind, yind = get_xy_inds(x,y,scale_factor)
    xoff, yoff = get_xy_offsets(x, y, xind, yind, scale_factor)
    wp, hp = get_parametrized_wh(w, h, scale_factor)
    if caffe_format:
        cls_vec = [cls] #classes are 1-4 (no zero on purpose, so that can be filtered out)
        objectness_vec = [1]
    else:
        cls_vec = convert_class_to_one_hot(cls, num_classes=num_classes)
        objectness_vec = [1, 0]
    box_loc = [xoff, yoff, wp, hp]
    box_vec = np.asarray(box_loc + objectness_vec + cls_vec)
    return box_vec

def test_grid(bbox, grid,kwargs):
    xdim, ydim, scale_factor,num_classes, caffe_format = kwargs["xdim"], kwargs["ydim"],kwargs["scale_factor"], kwargs["num_classes"],kwargs["caffe_format"]
    scale_factor = float(scale_factor)
    cls = int(bbox[4])
    x,y = bbox[0] / scale_factor, bbox[1] / scale_factor

    xo,yo = x - np.floor(x), y - np.floor(y)
    w,h = np.log2(bbox[2] / scale_factor), np.log2(bbox[3] / scale_factor)



    if caffe_format:
        depth = 6
        caffe_box = grid[:depth,int(x),int(y)]
        l_box = caffe_box
        lbl = [cls]
        obj = [1.]
    else:
        depth = 6 + num_classes
        oth_box = grid[int(x),int(y),:depth]
        l_box = oth_box
        obj = [1., 0.]
        

        lbl = num_classes*[0]
        lbl[cls-1] = 1
    
    real_box = [xo,yo,w,h] + obj
    real_box.extend(lbl)
    
    print l_box

    print real_box
    assert np.allclose(l_box, real_box), "Tests Failed"
#     if np.allclose(l_box, real_box) == True:
#         print "Yay! Passed Test"

def make_default_no_object_1hot(gr_truth):
    #make the 5th number 1, so the objectness by defualt is 0,1 -> denoting no object
    gr_truth[:,:,:,5] = 1.
    return gr_truth

