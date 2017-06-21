


# import sys
import sys
sys.path.append("..")
import numpy as np
import random
from matplotlib import patches
from matplotlib import pyplot as plt

from os.path import join
import os
classes = ["TD", "TC", "ETC", "AR"]
def plot_boxes(im,box_list_pred, box_list_gt):
    
    plt.figure(figsize=(10,10))
    
    sp = plt.subplot(111)
    sp.imshow(im,origin="lower")
    
    for box in box_list_pred:
        add_bbox(sp, box, color="r")
    
    for box in box_list_gt:
        add_bbox(sp, box, color="g")
    pass
    

def add_bbox(subplot, bbox, color):
        '''expects xcent,w,h boxes'''
        xcent,ycent,w,h = bbox[:4]
        xleft = xcent - w / 2.
        ybot = ycent - h / 2.
        cls = 0
        
        #flip x and y because matplotlib expects the vertical dimension to be x?
        #also xy is xmin,ymin -> bottom left corner
        subplot.add_patch(patches.Rectangle(xy=(ybot, xleft),
                                            width=h,
                                            height=w,
                                            lw=0.9,
                                            fill=False,
                                            color=color,alpha=1))
        
        #subplot.text(ymin+1,xmin,classes[cls], fontdict={"color":color })









