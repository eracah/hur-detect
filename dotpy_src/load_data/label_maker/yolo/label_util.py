


import sys
import numpy as np
import os
from os.path import join
if __name__ == "__main__":
    sys.path.append("../../../../")
    
from dotpy_src.load_data.configs import configs



def get_gr_truth_configs():
    tf_format = configs["tf_format"]
    one_hot = configs["one_hot_labels"]
    scale_factor = float(configs["scale_factor"])
    xdim, ydim = configs["input_shape"][-2:]
    num_classes = configs["num_classes"]
    
    #make sure xy coords divide cleanly with scale_factor
    assert xdim % scale_factor == 0 and ydim % scale_factor == 0,    "scale factor %i must divide the xy (%i, %i) coords cleanly " %(scale_factor,xdim, ydim)
    
    xlen, ylen = xdim / int(scale_factor), ydim / int(scale_factor)
    last_dim =  6 + num_classes if one_hot else 6 
    return scale_factor, xlen, ylen, last_dim, num_classes, one_hot, tf_format
    



def get_box_vector(coords, scale_factor, num_classes,one_hot):
    x,y,w,h,cls = coords
    xind, yind = get_xy_inds(x,y,scale_factor)
    xoff, yoff = get_xy_offsets(x, y, xind, yind, scale_factor)
    wp, hp = get_parametrized_wh(w, h, scale_factor)
    if one_hot:
        cls_vec = convert_class_to_one_hot(cls, num_classes=num_classes)
        objectness_vec = [1, 0]
    else:
        cls_vec = [cls] #classes are 1-4 (no zero on purpose, so that can be filtered out)
        objectness_vec = [1]

    box_loc = [xoff, yoff, wp, hp]
    box_vec = np.asarray(box_loc + objectness_vec + cls_vec)
    return box_vec



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
    ws , hs = w / scale_factor, h/ scale_factor
    wp, hp = np.log2(ws), np.log2(hs)
    return wp, hp
    



def convert_class_to_one_hot(class_num, num_classes):
    vec = num_classes * [0]
    vec[class_num ] = 1
    return vec

