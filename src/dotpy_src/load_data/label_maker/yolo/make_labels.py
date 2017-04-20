


import sys
import numpy as np
import time
import h5py
import os
from os.path import join
#from configs import configs


#put the path of where main.py runs on the path to simulate that setup for when we actually run
if __name__ == "__main__":
    sys.path.append("../../../../")
from label_util import get_gr_truth_configs, get_box_vector, get_xy_inds
from dotpy_src.load_data.label_maker.box_processing.coord_conversion import convert_min_max_to_wh_boxes



def make_labels(box_lists):
    '''takes a list of list of boxes, where box is [xmin,ymin, xmax,ymax,class?]'''
    label = _create_labels_from_example_box_list(box_lists[0])
    for ex_box_list in box_lists[1:]:
        box_tensor = _create_labels_from_example_box_list(box_lists[0])
        label = np.concatenate((label, box_tensor),axis=0)

    return label
        
    
    



def _create_labels_from_example_box_list(ex_box_list):
    
        # get some configs
        scale_factor, xlen, ylen, last_dim, num_classes, one_hot, tf_format = get_gr_truth_configs()
        
        boxes_tensor = np.zeros(( 1, xlen, ylen, last_dim ))

        ex_box_list = convert_min_max_to_wh_boxes(ex_box_list)


        for box in ex_box_list:
            x,y,w,h, cls = box
            xind, yind = get_xy_inds(x, y, scale_factor)
            box_vec = get_box_vector(box, scale_factor, num_classes,one_hot )
            boxes_tensor[:,xind, yind, :] = box_vec

        boxes_tensor = boxes_tensor if tf_format else np.transpose(boxes_tensor, axes=(0,3,1,2))
        return boxes_tensor
    



if __name__ == "__main__":

    box1 = [30,40,60,70,1]
    box2 = [2*b for b in box1]

    box_lists = [[box1,box2,box1],[box1,box2,box1],[box1,box2,box1],[box1,box2,box1]]
    print make_labels(box_lists).shape





