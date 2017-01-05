
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
import inspect
from os.path import join


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



#The etc files that use even time steps for labels:
#1979, 1980, 1982, 1983, 1984, 1985



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



def get_camfiles(data_dir, years):
    lsdir=listdir(data_dir)
    rpfile = re.compile(r"^cam5_.*\.nc$")
    camfiles = [f for f in lsdir if rpfile.match(f)]
    camfiles = [c for c in camfiles if get_timestamp(c).year in years]
    camfiles.sort()
    return camfiles
    



def normalize(arr,min_=None, max_=None, axis=(0,2,3)):
        if min_ is None or max_ is None:
            min_ = arr.min(axis=(0,2,3), keepdims=True)

            max_ = arr.max(axis=(0,2,3), keepdims=True)

        midrange = (max_ + min_) / 2.

        range_ = (max_ - min_) / 2.
        
        arr -= midrange

        arr /= (range_)
        return arr, min_, max_   



class BBoxIterator(object):
    def __init__(self,
                 years=[1979],
                 days=2,
                 batch_size = 1,
                 data_dir="/storeSSD/eracah/data/netcdf_ims/", 
                 metadata_dir="/storeSSD/eracah/data/metadata/",
                 shuffle=False, 
                 num_classes=4, 
                 labels_only=True, 
                 time_chunks_per_example=1,
                 no_labels_only=False,
                 time_stride=None, 
                 scale_factor=64., seed =5):
        
        frame = inspect.currentframe()
        self.set_data_members(frame)
        self.variables = [u'PRECT',u'PS',u'PSL',
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
        self.time_steps_per_day = 8
        self.xdim = 768
        self.ydim = 1152
        self.seed = seed
        self.camfiles = get_camfiles(self.data_dir, self.years)[:self.days]

            
    def set_data_members(self, frame):
        args, _, _, values = inspect.getargvalues(frame)
        del values['self']
        del values['frame']
        for k,v in values.iteritems():
            setattr(self,k,v)
            
    def iterate(self):
        for x,y in self.data_iterator():
#             print x.shape
#             print y.shape
            x, y = np.swapaxes(x, 1, 2), y
            y = y.astype("float32")
            if self.time_chunks_per_example == 1:
                x= np.squeeze(x,axis=2)
            if self.time_chunks_per_example > 1:
                y = np.squeeze(y,axis=0)
                
            
            yield x, y
            
    
    def data_iterator(self):
        '''
        Args:
           batch_size: number of examples in a batch
           data_dir: base dir where data is
           time_chunks_per_example: how many time steps are in a given example (default is one, but when we do 3D conv -> move to >1)
                                - should divide evenly into 8
        '''
        # for each day (out of 365 days)
        batch_size = self.batch_size
        for tensor, masks in self._day_iterator():  #tensor is 8,16,768,1152
            
            tensor, min_, max_ = normalize(tensor)
            time_chunks_per_day, variables, x,y  = tensor.shape #time_chunks will be 8
            assert time_chunks_per_day % self.time_chunks_per_example == 0, "For convenience,             the time chunk size should divide evenly for the number of time chunks in a single day"

            #reshapes the tensor into multiple spatiotemporal chunks of (chunk_size, 16, 768,1152)
            spatiotemporal_tensor = tensor.reshape(time_chunks_per_day / self.time_chunks_per_example, 
                                                   self.time_chunks_per_example, variables, x ,y)
            
            if self.time_chunks_per_example > 1:
                sp_mask = masks.reshape(self.time_steps_per_day / self.time_chunks_per_example, 
                                                   self.time_chunks_per_example / 2, 6 + self.num_classes, x /int(self.scale_factor) ,y / int(self.scale_factor))
            else:
                sp_mask = masks
            
            #if shuffle:
            #    np.random.shuffle(spatiotemporal_tensor)

            b = 0
            while True:
                if b*batch_size >= spatiotemporal_tensor.shape[0]:
                    break
                # todo: add labels

                yield spatiotemporal_tensor[b*batch_size:(b+1)*batch_size], sp_mask[b*batch_size:(b+1)*batch_size]
                b += 1
                
    def _day_iterator(self):
        """
        This iterator will return a pair of  tensors:
           * one is dimension (8, 16, 768, 1152) 
           * the other is dimension (8,12,18,9) 
                   -> 8 time steps, downsampled x, downsampled y, (xoffset, yoffset, w, h, confidence, softmax for 4 classes)
        each tensor corresponding to one of the 365 days of the year
        """
   
        # this directory can be accessed from cori

        
        
        if self.shuffle:
            np.random.RandomState(seed=self.seed).shuffle(self.camfiles)
        
        for camfile in self.camfiles:
            tr_data = self.grab_data(camfile) #one day slice per dataset
            masks = self.make_yolo_masks_for_dataset(camfile)
            
            
            if self.labels_only:
                # we assume labels are evn time steps here

                tr_data = tr_data[[0,2,4,6]]
            #masks are always only the labels!    
            masks = masks[[0,2,4,6]]
            
       

            yield tr_data, masks
            
            
#     def three_d_iterator(self, stride, time_steps_per_example):
#         camfiles = get_camfiles(self.data_dir, self.years)
#         camfiles = camfiles[:self.days]
#         cam_ind1 = 0
#         camfile1 = camfiles[0]
#         camfile2 = camfiles[1]
#         for start_step in range(0,time_steps_per_example, stride):
#             if start_step + stride <= self.time_steps_per_day:
                
#                 data = self.grab_data(camfile1, time_steps=range(start_step,start_step + stride))
#                 masks = self.make_yolo_masks_for_dataset(camfile1)
#                 y = masks[start_step: start_step + stride]
                
#             else:
#                 c1_time_steps = self.time_steps_per_day - start_step
#                 c2_time_steps = time_steps_per_example - c1_time_steps
#                 data1 = self.grab_data(camfile1, time_steps=range(start_step, self.time_steps_per_day))
#                 data2 = self.grab_data(camfile2, time_steps=range(0, c2_time_steps))
                
                
        
        
        
            
    
    
    def grab_data(self, filename, time_steps=range(8)):
        '''takes in: filename
           returns: (num_time_slices, num_variables,x,y) shaped tensor for day of data'''
        if len(time_steps) == 0:
            return
        dataset = nc.Dataset(join(self.data_dir, filename), "r", format="NETCDF4")
        data = [dataset[k][time_steps] for k in self.variables]
        tensor = np.vstack(data).reshape( len(self.variables),len(time_steps), self.xdim, self.ydim)
        tensor = np.swapaxes(tensor,0,1)
        return tensor


              
    def make_yolo_masks_for_dataset(self, camfile_name):

        labels_tensor = self.make_labels_for_dataset(camfile_name)
        labels_tensor = convert_bbox_minmax_to_cent_xywh(labels_tensor)


        yolo_mask = self.create_detection_gr_truth(bbox_tensor = labels_tensor)

        return yolo_mask
    
    
    def make_labels_for_dataset(self,fname):
        '''takes in string for fname and the number of time_steps and outputs
        a time_steps by maximages by 5 tensor encoding the coordinates and class of each event in a time step'''

        weather_types = ['tc','etc', 'us-ar']
        ts=get_timestamp(fname)
        maximagespertimestep=25

        # for every time step for every possible event, xmin,xmax,ymin,ymax,class
        bboxes = np.zeros((self.time_steps_per_day, maximagespertimestep, 5))
        event_counter = np.zeros((self.time_steps_per_day,))
        for weather_type in weather_types:
            selectdf = self.match_nc_to_csv(fname, weather_type)

            timelist=set(selectdf["time_step"])
            for t in timelist:
                t = int(t)

                coords_for_t = selectdf[selectdf["time_step"]==t].drop(["time_step"], axis=1).values
                coords_for_t = coords_for_t[(coords_for_t > 0).all(1)]

                # get current number of events and number of events for this time step
                num_events_for_t = coords_for_t.shape[0]
                cur_num_events = int(event_counter[t])

                #make slice
                slice_for_t = slice(cur_num_events, cur_num_events + num_events_for_t)

                #fill variables
                bboxes[t, slice_for_t] = coords_for_t
                event_counter[t] += num_events_for_t
        return bboxes
    
    
    def match_nc_to_csv(self, fname, weather_type, inc_csv=False):
        coord_keys = ["xmin", "xmax", "ymin", "ymax"]
        ts=get_timestamp(fname)

        if weather_type == 'us-ar':
            labeldf = pd.read_csv(join(self.metadata_dir, 'ar_labels.csv'))
            tmplabeldf=labeldf.ix[ (labeldf.month==ts.month) & (labeldf.day==ts.day) & (labeldf.year==ts.year) ].copy()
        else:
            labeldf = pd.read_csv(join(self.metadata_dir, '_'.join([str(ts.year),weather_type, 'labels.csv'])))
            tmplabeldf=labeldf.ix[ (labeldf.month==ts.month) & (labeldf.day==ts.day) ].copy()


        selectdf=tmplabeldf[["time_step"]+ coord_keys + ["category"]]
        if inc_csv is True:
            return selectdf, labeldf
        else:
            return selectdf 
        
        
    
    def create_detection_gr_truth(self,bbox_tensor):
        #x_xy : 1,2 tuple with x and y sizes for image
        #scale_factor: factor to scale xy size by fro gr_truth grid for YOLO
        #scale_factor = float(scale_factor)
        # xdim, ydim = 768,1152
        # scale_factor = 64
        # bbox_tensor = make_labels_for_dataset("cam5_1_amip_run2.cam2.h2.1984-01-03-00000.nc")
        # num_classes = 4 
        num_classes = self.num_classes
        scale_factor = float(self.scale_factor)
        bbox_classes = bbox_tensor[:,:,4]
        bbox_coords = bbox_tensor[:,:,:4]

        #make sure xy coords divide cleanly with scale_factor
        assert self.xdim % scale_factor == 0 and self.ydim % scale_factor == 0, "scale factor %i must divide the xy (%i, %i) coords cleanly " %(scale_factor,xdim, ydim)


        x_len,y_len = self.xdim / int(scale_factor), self.ydim / int(scale_factor)
        last_dim = 6 + num_classes #x,y,w,h,c plus num_classes for one hot encoding


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

    



        
        



# minw = 10000
# minh = 10000
# maxw= 0
# maxh = 0
# for weather_type in ["etc", "tc"]:
#     for year in range(1979,2006):
#         labeldf = pd.read_csv(join("/storeSSD/eracah/data/metadata", '_'.join([str(year),weather_type, 'labels.csv'])))
#         labeldf['widths'] = labeldf["xmax"] - labeldf["xmin"]
#         labeldf['heights'] = labeldf["ymax"] - labeldf["ymin"]
#         widths = labeldf['widths']
#         heights =  labeldf['heights']
#         if min(widths) < minw:
#             minw = min(widths)
            
#         if min(heights) < minh:
#             if min(heights) 
#             minh = min(heights)
            
#         if max(widths) > maxw:
#             maxw = min(widths)
            
#         if max(heights) > maxh:
#             if max(heights) == 454:
#                 break
#             print labeldf[['ymax', "ymin"]]
#             maxh = max(heights)
        

# print maxh, maxw, minh, minw
        
        

# f=labeldf[["ymin", "ymax", "heights", "category", "widths"]]

# f.ix[f["heights"] == 454]

# f=labeldf[["ymin", "ymax", "heights"]].sort_index

# f=labeldf[["ymin", "ymax", "heights"]].sort

# f

# maxw= 0
# maxh = 0
# for weather_type in ["us-ar"]:
#     for year in range(1979,2006):
#         labeldf = pd.read_csv(join("/storeSSD/eracah/data/metadata", 'ar_labels.csv'))
#         widths = labeldf["xmax"] - labeldf["xmin"]
#         heights = labeldf["ymax"] - labeldf["ymin"]
#         if max(widths) > maxw:
#             maxw = min(widths)
            
#         if max(heights) > maxh:
#             maxh = max(heights)

# maxw

# maxh

# !ls /storeSSD/eracah/data/metadata/




def count_events(year):
    metadata_dir = "/storeSSD/eracah/data/metadata/"
    coord_keys = ["xmin", "xmax", "ymin", "ymax"]
    d ={"us-ar":0, "etc":0, "tc":0}
    for weather_type in d.keys(): 
        if weather_type == 'us-ar':
            labeldf = pd.read_csv(join(metadata_dir, 'ar_labels.csv'))
            labeldf = labeldf.ix[(labeldf.year==year)]
            d[weather_type] = len(labeldf)
#             tmplabeldf=labeldf.ix[ (labeldf.month==ts.month) & (labeldf.day==ts.day) & (labeldf.year==ts.year) ].copy()
        else:
            labeldf = pd.read_csv(join(metadata_dir, '_'.join([str(year),weather_type, 'labels.csv'])))
#             tmplabeldf=labeldf.ix[ (labeldf.month==ts.month) & (labeldf.day==ts.day) ].copy()
            if weather_type == "etc":
                d[weather_type] = len(labeldf)
            else:
                d["td"] = len(labeldf[labeldf["str_category"] == "tropical_depression"])
                d["tc"] = len(labeldf[labeldf["str_category"] == "tropical_cyclone"])
            
    return d



def count_events_md(year, last_md = (3,16)):
    lm, ld = last_md
    metadata_dir = "/storeSSD/eracah/data/metadata/"
    coord_keys = ["xmin", "xmax", "ymin", "ymax"]
    d ={"us-ar":0, "etc":0, "tc":0}
    for weather_type in d.keys(): 
        if weather_type == 'us-ar':
            labeldf = pd.read_csv(join(metadata_dir, 'ar_labels.csv'))
            for i in range(1,lm):
                labeldf = labeldf.ix[(labeldf.year==year) & (labeldf.month == i)]
                d[weather_type] += len(labeldf)
            for i in range(1,ld+1):
                labeldf_end = labeldf.ix[(labeldf.year==year) & (labeldf.month ==lm) & (labeldf.day == i)]
                d[weather_type] += len(labeldf_end)
           
#             tmplabeldf=labeldf.ix[ (labeldf.month==ts.month) & (labeldf.day==ts.day) & (labeldf.year==ts.year) ].copy()
        else:
            labeldf = pd.read_csv(join(metadata_dir, '_'.join([str(year),weather_type, 'labels.csv'])))
            if weather_type == "etc":
                for i in range(1,lm):
                    labeldfi = labeldf.ix[(labeldf.year==year) & (labeldf.month == i)]
                    d[weather_type] += len(labeldfi)
                for i in range(1,ld+1):
                    labeldfi = labeldf.ix[(labeldf.year==year) & (labeldf.month ==lm) & (labeldf.day == i)]
                    d[weather_type] += len(labeldfi)
            else:
                d["td"] = 0
                for i in range(1,lm):
                    labeldfi = labeldf.ix[(labeldf.year==year) & (labeldf.month == i)]
                    d["td"] += len(labeldfi[labeldfi["str_category"] == "tropical_depression"])
                    d["tc"] += len(labeldfi[labeldfi["str_category"] == "tropical_cyclone"])
                for i in range(1,ld+1):
                    labeldfi = labeldf.ix[(labeldf.year==year) & (labeldf.month ==lm) & (labeldf.day == i)]
                    d["td"] += len(labeldfi[labeldfi["str_category"] == "tropical_depression"])
                    d["tc"] += len(labeldfi[labeldfi["str_category"] == "tropical_cyclone"])
              
    return d







def get_percents(event_dict):
    tot = sum(event_dict.values())
    new_d = {k: float(v)/ tot for k,v in event_dict.iteritems()}
    return new_d



get_percents(count_events(1979))



count_events(1984)



count_events_md(1982)



count_events_md(1985)



get_percents(count_events(1984))



365 * 8



metadata_dir = "/storeSSD/eracah/data/metadata/"



labeldf = pd.read_csv(join(metadata_dir, '_'.join([str(1979),"tc", 'labels.csv'])))



len(labeldf[labeldf["str_category"] == "tropical_depression"])

len(labeldf[labeldf["str_category"] == "tropical_cyclone"])



range(1,3)





