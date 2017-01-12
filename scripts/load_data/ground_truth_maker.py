
import matplotlib; matplotlib.use("agg")


import sys
import numpy as np
from util import convert_bbox_minmax_to_cent_xywh
import time
from label_loader import  make_labels_for_dataset



def make_yolo_masks_for_dataset( camfile_name, kwargs):
        t = time.time()
        labels_tensor = make_labels_for_dataset(camfile_name,kwargs)
        labels_tensor = convert_bbox_minmax_to_cent_xywh(labels_tensor)


        yolo_mask = create_detection_gr_truth(labels_tensor, kwargs)
        print "make gt masks: ", time.time() -t
        return yolo_mask



def create_detection_gr_truth(bbox_tensor,kwargs):
        #x_xy : 1,2 tuple with x and y sizes for image
        #scale_factor: factor to scale xy size by fro gr_truth grid for YOLO
        #scale_factor = float(scale_factor)
        # xdim, ydim = 768,1152
        # scale_factor = 64
        # bbox_tensor = make_labels_for_dataset("cam5_1_amip_run2.cam2.h2.1984-01-03-00000.nc")
        # num_classes = 4 
        scale_factor = float(kwargs["scale_factor"])
        bbox_classes = bbox_tensor[:,:,4]
        bbox_coords = bbox_tensor[:,:,:4]
        xdim,ydim = kwargs["xdim"], kwargs["ydim"]
        
        #make sure xy coords divide cleanly with scale_factor
        assert xdim % scale_factor == 0 and ydim % scale_factor == 0, "scale factor %i must divide the xy (%i, %i) coords cleanly " %(scale_factor,xdim, ydim)


        x_len,y_len = xdim / int(scale_factor), ydim / int(scale_factor)
        last_dim = 6 + kwargs["num_classes"] #x,y,w,h,conf1,conf2 plus num_classes for one hot encoding


        #divide up bbox with has range 0-95 to 0-95/scale_factor (so 6x6 for scale factor of 16)
        bb_scaled = bbox_coords / scale_factor


        #each coordinate goes at index i,j in the 6x6 array, where i,j are the coordinates of the
        #lower left corner of the grid that center of the box (in 6x6 space ) falls on
        #subtract eps so we dont't have one off error
        eps = np.finfo(float).eps
        inds = np.floor(bb_scaled[:,:,:2]-10*eps).astype('int')

        #xywh where x and y are offset from lower left corner of grid thay are in [0,1] and w and h
        # are what fraction the width and height of bboxes are of the total width and total height of the image
        xywh = np.copy(bb_scaled)

        #subtract the floored values to get the offset from the grid cell
        xywh[:,:,:2] -= inds[:,:,:2].astype('float')


        #divide by scaled width and height to get wdith and height relative to width and height of box
        xywh[:,:,2] = np.log2(bbox_coords[:,:,2] / scale_factor)
        xywh[:,:,3] = np.log2(bbox_coords[:,:,3] / scale_factor)


        #make gr_truth which is 

        gr_truth = np.zeros((bbox_coords.shape[0],last_dim, x_len, y_len ))
    #     else:
    #         gr_truth = np.zeros((bbox_coords.shape[0], x_len,y_len,last_dim))


        #sickens me to a do a for loop here, but numpy ain't cooperating
        # I tried gr_truth[np.arange(gr_truth.shape[0]),inds[:0], inds[:1]][:,4] = xywh
        #but it did not work

        # we assume one box per image here
        # for each grid point that is center of image plop in center, and width and height and class
        for i in range(gr_truth.shape[0]):
            #put coordinates, conf and class for all events (now there are multiple)
            for j, coords in enumerate(xywh[i]):


                # the index into the groudn truth grid where class should go
                xind, yind = inds[i,j,0], inds[i,j,1]
                gr_truth[i, :4, xind,yind,] = coords

                #put in confidence
                gr_truth[i,4,xind,yind] = 1 if bbox_classes[i,j] > 0. else 0.
                gr_truth[i,5,xind,yind] = 1 if gr_truth[i,4,xind,yind] == 0. else 0.
                #put in class label
                gr_truth[i, 5 + int(bbox_classes[i,j]),xind,yind] = 1. if bbox_classes[i,j] > 0. else 0.

        return gr_truth







def test_grid(bbox, grid, xdim, ydim, scale_factor,num_classes, caffe_format=False):
    cls = int(bbox[4])
    x,y = bbox[0] / scale_factor, bbox[1] / scale_factor
    xo,yo = (bbox[0] % scale_factor) / float(scale_factor), (bbox[1] % scale_factor) / float(scale_factor)
    w,h = bbox[2] / scale_factor / (xdim / scale_factor), bbox[3] / scale_factor/ (ydim / scale_factor)
    
    depth = 5 + num_classes
    if caffe_format:
        l_box = grid[:depth,x,y]
    else:
        l_box = grid[int(x),int(y),:depth]
    lbl = num_classes*[0]
    lbl[cls-1] = 1
    
    real_box = [xo,yo,w,h,1.]
    real_box.extend(lbl)
    
    print l_box
    print real_box
    assert np.allclose(l_box, real_box), "Tests Failed"

