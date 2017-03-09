


import sys
import numpy as np
import time
from label_loader import  make_labels_for_dataset
import h5py
import os
from os.path import join
from util import *
from configs import configs




def test(camfile_path, mask, kwargs):
    labels_csv_file = join(kwargs["metadata_dir"], "labels.csv")
    box_list = make_labels_for_dataset(camfile_path, labels_csv_file)
    box = box_list[0][0]
    test_grid(box, mask[0], kwargs)



def create_yolo_gr_truth(bbox_list, year, caffe_format=False):

        scale_factor, xdim, ydim, num_classes = configs["scale_factor"], configs["xdim"], configs["ydim"], configs["num_classes"] 
        
        xlen, ylen = xdim /scale_factor, ydim / scale_factor
        if caffe_format:
            last_dim = 6  #(xywh obj cls)
        else:
            last_dim = 6 + num_classes

        num_time_steps = len(bbox_list)
        
        gr_truth = np.zeros(( num_time_steps, xlen, ylen, last_dim ))

        # if the file is specified as unlabelled then we skip this
        # and return gr_truth as is -> all zeros

        if not caffe_format:
            gr_truth = make_default_no_object_1hot(gr_truth)
        
        
        # for caffe we have the channels as the following x,y,w,h,obj,cls
        # obj is 1 or 0 and cls is 1-4 if an obj is there and 0 if not
        #For noncaffe we have x,y,w,h,obj,no-obj, cls1,cls2,cls3,cls4
        #cls1-cls4 is one hot encoded vector
        for time_step in range(num_time_steps):
            for coords in bbox_list[time_step]:
                x,y,w,h,cls = coords

                xind, yind = get_xy_inds(x,y,scale_factor)
                box_vec = get_box_vector(coords, scale_factor, num_classes, caffe_format)
                gr_truth[time_step,xind,yind,:] = box_vec

        if caffe_format:
            gr_truth = np.transpose(gr_truth, axes=(0,3,1,2))
        return gr_truth



def make_yolo_masks_for_dataset( camfile_path,labels_csv_file, caffe_format=False):
        ts = get_timestamp(camfile_path)
        box_list = make_labels_for_dataset(camfile_path, labels_csv_file)
        yolo_mask = create_yolo_gr_truth(box_list,ts.year, caffe_format)
        return yolo_mask



def make_multiple_yolo_masks(camfile_paths,labels_csv_file, caffe_format):
    ym = make_yolo_masks_for_dataset(camfile_paths[0], labels_csv_file, caffe_format)
    for camfile_path in camfile_paths[1:]:
        tmp = make_yolo_masks_for_dataset(camfile_path, labels_csv_file, caffe_format)
        ym = np.vstack((ym,tmp))
    return ym



if __name__ == "__main__":
    ym=make_yolo_masks_for_dataset( camfile_path="/home/evan/data/climate/input/cam5_1_amip_run2.cam2.h2.1979-01-06-00000.nc",
                                labels_csv_file="/home/evan/data/climate/labels/labels.csv",
                                caffe_format=True)
    yms = make_multiple_yolo_masks(camfile_paths=
                                   ["/home/evan/data/climate/input/cam5_1_amip_run2.cam2.h2.1979-01-06-00000.nc",
                                    "/home/evan/data/climate/input/cam5_1_amip_run2.cam2.h2.1979-01-08-00000.nc",
                                    "/home/evan/data/climate/input/cam5_1_amip_run2.cam2.h2.1979-01-07-00000.nc"],
                                    labels_csv_file="/home/evan/data/climate/labels/labels.csv",
                                    caffe_format=True)
    
    
    





