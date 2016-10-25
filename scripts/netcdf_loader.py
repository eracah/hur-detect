
import matplotlib; matplotlib.use("agg")


import netCDF4 as nc
from os import listdir, system
from os.path import isfile, join, isdir
import re
import numpy as np
from shutil import copyfile
import imp
import itertools
from sklearn.manifold import TSNE
import numpy as np
import cPickle as pickle
import gzip
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import sys
from time import time
sys.path.append("..")
from pylab import rcParams
rcParams['figure.figsize'] = 15, 20
import pdb
import itertools
import datetime as dt
import pandas as pd


# If you view a map:
#  * longitude: horizontal
#  * latitude: vertical
#  
# Defined Here:
#  * longitude (horiz): y
#  * latitude (vertical): x
#  
# Array-wise:
#  * dim1(x) : vertical
#  * dim2(y) : horizontal
#  
# So:
#  * dim1 of array is latitude (thus x)
#  * dim2 is longitude (thus y)
# 
# So if we define something as xmin,xmax,ymin,ymax here:
#  * filling in that box in the array is:
#     * arr[xmin:xmax, ymin:ymax] = 0
#     
#     
# * and the array is 768,1152 ?
# 
# 
# 
# #LABEL NUMBERS
# * Tropical Depression is 1
# * Hurricane is 2
# * ETC is 3
# * AR is 4


#The etc files that use even time steps for labels:
#1979, 1980, 1982, 1983, 1984, 1985



# gbdl = "/project/projectdirs/dasrepo/gordon_bell/deep_learning/"
# imdir = gbdl + "data/climate/CAM5_0.25/climo/big_images/"

gbdl = "/storeSSD/eracah/nersc/data/"
imdir = "/storeSSD/cbeckham/nersc/big_images/1979/"
#ds = nc.Dataset(join(imdir,"netcdf_files", "cam5_1_amip_run2.cam2.h2.1984-09-06-00000.nc"))
metadatadir = join(gbdl, "teca_metadata")



def make_labels_for_dataset(fname, metadata_dir, time_steps=8):
    '''takes in string for fname and the number of time_steps and outputs
    a time_steps by maximages by 5 tensor encoding the coordinates and class of each event in a time step'''
   
    weather_types = ['tc','etc', 'us-ar']
    ts=get_timestamp(fname)
    maximagespertimestep=25
    
    # for every time step for every possible event, xmin,xmax,ymin,ymax,class
    bboxes = np.zeros((time_steps, maximagespertimestep, 5))
    event_counter = np.zeros((time_steps,))
    for weather_type in weather_types:
        selectdf = match_nc_to_csv(fname, metadata_dir, weather_type)
    
        timelist=set(selectdf["time_step"])
        for t in timelist:
            t = int(t)

            coords_for_t = selectdf[selectdf["time_step"]==t].drop(["time_step"], axis=1).values
            
            # get current number of events and number of events for this time step
            num_events_for_t = coords_for_t.shape[0]
            cur_num_events = int(event_counter[t])
            
            #make slice
            slice_for_t = slice(cur_num_events, cur_num_events + num_events_for_t)

            #fill variables
            bboxes[t, slice_for_t] = coords_for_t
            event_counter[t] += num_events_for_t
    return bboxes



def match_nc_to_csv(fname,metadata_dir,weather_type, inc_csv=False):
    coord_keys = ["xmin", "xmax", "ymin", "ymax"]
    ts=get_timestamp(fname)

    if weather_type == 'us-ar':
        labeldf = pd.read_csv(join(metadata_dir, 'ar_labels.csv'))
        tmplabeldf=labeldf.ix[ (labeldf.month==ts.month) & (labeldf.day==ts.day) & (labeldf.year==ts.year) ].copy()
    else:
        labeldf = pd.read_csv(join(metadata_dir, '_'.join([str(ts.year),weather_type, 'labels.csv'])))
        tmplabeldf=labeldf.ix[ (labeldf.month==ts.month) & (labeldf.day==ts.day) ].copy()
    
    
    selectdf=tmplabeldf[["time_step"]+ coord_keys + ["category"]]
    if inc_csv is True:
        return selectdf, labeldf
    else:
        return selectdf 
            



def get_timestamp(filename):
    rpyear = re.compile(r"(\.h2\.)(.*?)(-)")
    rpdaymonth = re.compile(r"(-)(.*?)(\d{5}\.)")
    year=int(rpyear.search(filename).groups()[1])
    tmp=rpdaymonth.search(filename).groups()[1].split('-')
    month=int(tmp[0])
    day=int(tmp[1])
    return dt.date(year,month,day)



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



#convert_bbox_minmax_to_cent_xywh(make_labels_for_dataset("cam5_1_amip_run2.cam2.h2.1984-01-03-00000.nc"))



# bbox_t = convert_bbox_minmax_to_cent_xywh(make_labels_for_dataset("cam5_1_amip_run2.cam2.h2.1984-01-03-00000.nc"))

# grid_t = make_yolo_masks_for_dataset("cam5_1_amip_run2.cam2.h2.1984-01-03-00000.nc")

# bbox =bbox_t[0,2]
# grid = grid_t[0]

# test_grid(bbox,grid,768,1152,64,4)



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



def create_detection_gr_truth(xdim, ydim, scale_factor, bbox_tensor, num_classes):
    #x_xy : 1,2 tuple with x and y sizes for image
    #scale_factor: factor to scale xy size by fro gr_truth grid for YOLO
    #scale_factor = float(scale_factor)
    # xdim, ydim = 768,1152
    # scale_factor = 64
    # bbox_tensor = make_labels_for_dataset("cam5_1_amip_run2.cam2.h2.1984-01-03-00000.nc")
    # num_classes = 4 

    scale_factor = float(scale_factor)
    bbox_classes = bbox_tensor[:,:,4]
    bbox_coords = bbox_tensor[:,:,:4]

    #make sure xy coords divide cleanly with scale_factor
    assert xdim % scale_factor == 0 and ydim % scale_factor == 0, "scale factor %i must divide the xy (%i, %i) coords cleanly " %(scale_factor,xdim, ydim)


    x_len,y_len = xdim / int(scale_factor), ydim / int(scale_factor)
    last_dim = 5 + num_classes #x,y,w,h,c plus num_classes for one hot encoding


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


    #divide by scaled width and height to get wdith and height relative to width and height of image (width is just xrange, height is yrange)
    xywh[:,:,2] /= x_len
    xywh[:,:,3] /= y_len


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

            #put in class label
            gr_truth[i, 4 + int(bbox_classes[i,j]),xind,yind] = 1. if bbox_classes[i,j] > 0. else 0.

    return gr_truth




def make_yolo_masks_for_dataset(camfile_name,metadata_dir="/storeSSD/eracah/data/teca_metadata/", xdim=768, ydim=1152,time_steps=8, classes=4):
    
    labels_tensor = make_labels_for_dataset(camfile_name,metadata_dir, time_steps)
    labels_tensor = convert_bbox_minmax_to_cent_xywh(labels_tensor)


    yolo_mask = create_detection_gr_truth(xdim,ydim,scale_factor=64.,bbox_tensor = labels_tensor, num_classes=classes)
    
    #masks is an 8,num_classes, 768, 1152 mask 0's everywhere except where class is
    return yolo_mask



# match_nc_to_csv("cam5_1_amip_run2.cam2.h2.1984-09-06-00000.nc", "us-ar", inc_csv=True)[1]



def make_time_slice(dataset, time, variables, x=768, y=1152):
    '''Takes in a dataset, a time and variables and gets one time slice for all the variables and all x and y'''
    variables_at_time_slice = [dataset[k][time] for k in variables]
    tensor = np.vstack(variables_at_time_slice).reshape(len(variables), x,y)
    
    return tensor


def make_spatiotemporal_tensor(dataset,num_time_slices, variables, x=768, y=1152):
    '''takes in: dataset, num_time_slices
       returns: num_time_slices, num_variables,x,y'''
    time_slices = [ make_time_slice(dataset, time, variables) for time in range(num_time_slices) ]
    tensor = np.vstack(time_slices).reshape(num_time_slices, len(variables), x, y)

    return tensor




def _day_iterator(years=[1979], metadata_dir="/storeSSD/eracah/data/teca_metadata/", data_dir="/project/projectdirs/dasrepo/gordon_bell/climate/data/big_images/",
                  shuffle=False, days=365, time_steps=8, classes=2, labels_only=True):
    """
    This iterator will return a pair of  tensors:
       * one is dimension (8, 16, 768, 1152) 
       * the other is dimension (8,12,18,9) 
               -> 8 time steps, downsampled x, downsampled y, (xoffset, yoffset, w, h, confidence, softmax for 4 classes)
    each tensor corresponding to one of the 365 days of the year
    """
    variables = [u'PRECT',
                 u'PS',
                 u'PSL',
                 u'QREFHT',
                 u'T200',
                 u'T500',
                 u'TMQ',
                 u'TREFHT',
                 u'TS',
                 u'U850',
                 u'UBOT',
                 u'V850',
                 u'VBOT',
                 u'Z1000',
                 u'Z200',
                 u'ZBOT']    
    # this directory can be accessed from cori
    maindir = data_dir #+ str(year) 
    lsdir=listdir(maindir)
    rpfile = re.compile(r"^cam5_.*\.nc$")
    camfiles = [f for f in lsdir if rpfile.match(f)]
    camfiles = [c for c in camfiles if get_timestamp(c).year in years]
    
    camfiles.sort()
    camfiles = camfiles[:days]
    
    if shuffle:
        np.random.shuffle(camfiles)
        #sys.stderr.write("warning: shuffling camfiles in _day_iterator()\n")
    for camfile in camfiles:
        dataset = nc.Dataset(maindir+'/'+camfile, "r", format="NETCDF4")
        x=768
        y=1152
        day_slice = make_spatiotemporal_tensor(dataset,time_steps,variables) #one day slice per dataset
        tr_data = day_slice.reshape(time_steps,len(variables), x, y)
        masks = make_yolo_masks_for_dataset(camfile,metadata_dir=metadata_dir, xdim=768, ydim=1152,time_steps=8, classes=classes)
        if labels_only:
            # we assume labels are evn time steps here
            tr_data = tr_data[[0,2,4,6]]
            masks = masks[[0,2,4,6]]
        
        yield tr_data, masks




def data_iterator(batch_size,
                  data_dir="/project/projectdirs/dasrepo/gordon_bell/deep_learning/data/climate/big_images/",
                  metadata_dir="/storeSSD/eracah/data/teca_metadata/",
                  time_chunks_per_example=8,
                  shuffle=False,
                  days=365,
                  years=[1979],
                  time_steps=8, classes=2, labels_only=True):
    '''
    Args:
       batch_size: number of examples in a batch
       data_dir: base dir where data is
       time_chunks_per_example: how many time steps are in a given example (default is one, but when we do 3D conv -> move to >1)
                            - should divide evenly into 8
    '''
    # for each day (out of 365 days)
    day=0
    for tensor, masks in _day_iterator(years=years,data_dir=data_dir,metadata_dir=metadata_dir,
                                       shuffle=shuffle,classes=classes,days=days, 
                                       time_steps=time_steps,
                                       labels_only=labels_only):  #tensor is 8,16,768,1152
        # preprocess for day
        tensor, min_, max_ = normalize(tensor)
        
        #TODO: preprocess over everything
        #TODO: split up into train,test, val
        time_chunks_per_day, variables, h, w = tensor.shape #time_chunks will be 8
        assert time_chunks_per_day % time_chunks_per_example == 0, "For convenience,         the time chunk size should divide evenly for the number of time chunks in a single day"
        
        #reshapes the tensor into multiple spatiotemporal chunks of (chunk_size, 16, 768,1152)
        spatiotemporal_tensor = tensor.reshape(time_chunks_per_day / time_chunks_per_example, 
                                               time_chunks_per_example, variables, h ,w)

        sp_mask = masks
        sp_mask = masks.reshape(time_chunks_per_day / time_chunks_per_example, time_chunks_per_example,*masks.shape[1:])
        
        #if shuffle:
        #    np.random.shuffle(spatiotemporal_tensor)

        b = 0
        while True:
            if b*batch_size >= spatiotemporal_tensor.shape[0]:
                break
            # todo: add labels

            yield spatiotemporal_tensor[b*batch_size:(b+1)*batch_size], sp_mask[b*batch_size:(b+1)*batch_size]
            b += 1



def normalize(arr,min_=None, max_=None, axis=(0,2,3)):
        if min_ is None or max_ is None:
            min_ = arr.min(axis=(0,2,3), keepdims=True)

            max_ = arr.max(axis=(0,2,3), keepdims=True)

        midrange = (max_ + min_) / 2.

        range_ = max_ - min_

        arr = (arr - midrange) / (range_ /2.)
        return arr, min_, max_   



# -------------------------------------

#tropical depression are 0
# hurricanes are 1


def bbox_iterator(years,days,
                  batch_size = 1,
                  data_dir="/storeSSD/eracah/data/netcdf_ims/", 
                  metadata_dir="/storeSSD/eracah/data/metadata/",
                shuffle=False, num_classes=4, labels_only=True, time_chunks_per_example=1 ):
    
    """years: list of years,
        days: number of days
        classes: number of classes
        labels_only: -> if true -> only does images with labels"""
    for x,y in data_iterator(years=years,batch_size=batch_size, data_dir=data_dir, metadata_dir=metadata_dir, time_chunks_per_example=time_chunks_per_example,
                  shuffle=shuffle,days=days, classes=num_classes, labels_only=labels_only):

            x, y = np.swapaxes(x, 1, 2), y
            y = y.astype("float32")
            yield x, y
            



def convert_coord_tens_to_box(coord_tens, xind, yind, scale_factor, xdim=768,ydim=1152):
    
    
    xoff,yoff,w,h = coord_tens
    
    x,y = xind+ xoff, yind+ yoff
    
    x,y,w,h = [scale_factor * c for c in [x,y,(xdim/scale_factor)*w,(ydim/scale_factor)*h] ]
    
    return x,y,w,h



if __name__ == '__main__':
    dir_kwargs = dict(data_dir="/storeSSD/eracah/data/netcdf_ims/", metadata_dir="/storeSSD/eracah/data/metadata")
    tr_kwargs = dict(years=[1979], days=5)
    tr_kwargs.update(dir_kwargs)
    val_kwargs= dict(years=[1979], days=2)
    val_kwargs.update(dir_kwargs)
    print tr_kwargs
    for x,y in bbox_iterator(**tr_kwargs):
        print x.shape, y.shape
        
    
        

