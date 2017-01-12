
import matplotlib; matplotlib.use("agg")


import re
import numpy as np
import datetime as dt
import os



def normalize(arr,min_=None, max_=None, axis=(0,2,3)):
        if min_ is None or max_ is None:
            min_ = arr.min(axis=(0,2,3), keepdims=True)

            max_ = arr.max(axis=(0,2,3), keepdims=True)

        midrange = (max_ + min_) / 2.

        range_ = (max_ - min_) / 2.
        
        arr -= midrange

        arr /= (range_)
        return arr, min_, max_   




def convert_bbox_minmax_to_cent_xywh(bboxes):
    #current bbox set up is xmin,ymin,xmax,ymax
    xmin, xmax,ymin,  ymax = [ bboxes[:,:,i] for i in range(4) ]
    
    w = xmax - xmin
    h = ymax - ymin

    x_c = xmin + w / 2.
    y_c = ymin + h / 2.
    
    
    bboxes[:,:,0] = x_c
    bboxes[:,:,1] = y_c
    bboxes[:,:,2] = w # w
    bboxes[:,:,3] = h #h
    return bboxes



def get_timestamp(filename):
    rpyear = re.compile(r"(\.h2\.)(.*?)(-)")
    rpdaymonth = re.compile(r"(-)(.*?)(\d{5}\.)")
    year=int(rpyear.search(filename).groups()[1])
    tmp=rpdaymonth.search(filename).groups()[1].split('-')
    month=int(tmp[0])
    day=int(tmp[1])
    return dt.date(year,month,day)



def get_camfiles(data_dir, years):
    lsdir=os.listdir(data_dir)
    rpfile = re.compile(r"^cam5_.*\.nc$")
    camfiles = [f for f in lsdir if rpfile.match(f)]
    camfiles = [c for c in camfiles if get_timestamp(c).year in years]
    camfiles.sort()
    return camfiles

