
import matplotlib; matplotlib.use("agg")


import sys
import numpy as np
import time
from label_loader import  make_labels_for_dataset



def make_yolo_masks_for_dataset( camfile_name, kwargs, labels_csv_file):
        box_list = make_labels_for_dataset(camfile_name, labels_csv_file)
        yolo_mask = create_yolo_gr_truth(box_list, kwargs)
        return yolo_mask



def get_gr_truth_configs(kwargs):
    scale_factor = float(kwargs["scale_factor"])
    xdim, ydim = kwargs["xdim"], kwargs["ydim"]
    num_classes = kwargs["num_classes"]
    #make sure xy coords divide cleanly with scale_factor
    assert xdim % scale_factor == 0 and ydim % scale_factor == 0, "scale factor %i must divide the xy (%i, %i) coords cleanly " %(scale_factor,xdim, ydim)
    
    xlen, ylen = xdim / int(scale_factor), ydim / int(scale_factor)
    
    #x,y,w,h,conf1,conf2 plus num_classes for one hot encoding
    last_dim = 6 + num_classes
    return scale_factor, xlen, ylen, last_dim, num_classes 
    



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
    vec[class_num - 1] = 1
    return vec



def get_box_vector(coords, scale_factor, num_classes):
    x,y,w,h,cls = coords
    xind, yind = get_xy_inds(x,y,scale_factor)
    xoff, yoff = get_xy_offsets(x, y, xind, yind, scale_factor)
    wp, hp = get_parametrized_wh(w, h, scale_factor)
    class_1hot_vec = convert_class_to_one_hot(cls, num_classes=num_classes)
    objectness = [1, 0]
    box_loc = [xoff, yoff, wp, hp]
    box_vec = np.asarray(box_loc + objectness + class_1hot_vec)
    return box_vec



def make_default_no_object_1hot(gr_truth):
    #make the 5th number 1, so the objectness by defualt is 0,1 -> denoting no object
    gr_truth[:,:,:,5] = 1.
    return gr_truth



def create_yolo_gr_truth(bbox_list, kwargs, caffe_format=False):
        scale_factor, xlen, ylen, last_dim, num_classes = get_gr_truth_configs(kwargs)
        
        num_time_steps = len(bbox_list)
        
        gr_truth = np.zeros(( num_time_steps, xlen, ylen, last_dim ))
        gr_truth = make_default_no_object_1hot(gr_truth)
        
        
        
        
        for time_step in range(num_time_steps):
            for coords in bbox_list[time_step]:
                x,y,w,h,cls = coords

                xind, yind = get_xy_inds(x,y,scale_factor)
                box_vec = get_box_vector(coords, scale_factor, num_classes)
                gr_truth[time_step,xind,yind,:] = box_vec

        if caffe_format:
            gr_truth = np.transpose(gr_truth, axes=(0,3,1,2))
        return gr_truth



def test_grid(bbox, grid, xdim, ydim, scale_factor,num_classes, caffe_format=False):
    scale_factor = float(scale_factor)
    cls = int(bbox[4])
    x,y = bbox[0] / scale_factor, bbox[1] / scale_factor

    xo,yo = x - np.floor(x), y - np.floor(y)
    w,h = np.log2(bbox[2] / scale_factor), np.log2(bbox[3] / scale_factor)


    depth = 6 + num_classes
    if caffe_format:
        l_box = grid[:depth,x,y]
    else:
        l_box = grid[int(x),int(y),:depth]

    lbl = num_classes*[0]
    lbl[cls-1] = 1
    
    real_box = [xo,yo,w,h,1.,0.]
    real_box.extend(lbl)
    
    print l_box
    print real_box
    assert np.allclose(l_box, real_box), "Tests Failed"



if __name__ == "__main__":
    kwargs = {  "metadata_dir": "/home/evan/data/climate/labels/",
                "scale_factor": 64, 
                "xdim":768,
                "ydim":1152,
                "time_steps_per_file": 8,
                "num_classes": 4, }

    camfile_name = "/home/evan/data/climate/input/cam5_1_amip_run2.cam2.h2.1979-01-05-00000.nc"
    labels_csv_file = "/home/evan/data/climate/labels/labels.csv"
    ym = make_yolo_masks_for_dataset(camfile_name,
                                     kwargs,
                                    labels_csv_file)
    box_list = make_labels_for_dataset(camfile_name, labels_csv_file)
    box = box_list[0][0]
    box_vec = get_box_vector(box, scale_factor=64., num_classes=4)
    test_grid(box, ym[0], kwargs["xdim"], kwargs["ydim"], kwargs["scale_factor"], kwargs["num_classes"])

