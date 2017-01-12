
import matplotlib; matplotlib.use("agg")


import sys
import os

import netCDF4 as nc
from netCDF4 import MFDataset
from os import listdir, system
from os.path import isfile, join, isdir
import numpy as np
import imp
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import time
import inspect
from util import get_camfiles, normalize
from ground_truth_maker import make_yolo_masks_for_dataset



class BBoxIterator(object):
    def __init__(self,kwargs,mode="tr"):
        self.kwargs = kwargs
        #frame = inspect.currentframe()
        self.set_data_members(kwargs)
        self.seed = 7
        self.camfiles = get_camfiles(self.data_dir, self.kwargs[mode+ "_years"])[:self.kwargs[mode + "_days"]]
        self.ind = 0
        
            
    def set_data_members(self, kwargs):
        #args, _, _, values = inspect.getargvalues(frame)
        for k,v in kwargs.iteritems():
            setattr(self,k,v)
            
    def iterate(self):
        for x,y in self.data_iterator():
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
            masks = make_yolo_masks_for_dataset(camfile, kwargs)
            
            
            #masks are always only the labels!    
            masks = masks[[0,2,4,6]]
            
       

            yield tr_data, masks
            

                   
    
    def grab_data(self, filename, time_steps=[0,2,4,6]):
        '''takes in: filename
           returns: (num_time_slices, num_variables,x,y) shaped tensor for day of data'''
        t = time.time()
        if len(time_steps) == 0:
            return
        dataset = nc.Dataset(join(self.data_dir, filename), "r", format="NETCDF4")
        data = [dataset[k][time_steps] for k in self.variables]
        tensor = np.vstack(data).reshape( len(self.variables),len(time_steps), self.xdim, self.ydim)
        tensor = np.swapaxes(tensor,0,1)
        print "initial io and vstack: ", time.time() -t 
        return tensor
    
    
    def get_next_chunk(self):
        mfo = kwargs["max_files_open"]
        
        #if we are starting back up again shuffle everything
        if self.ind < mfo:
            if self.kwargs["shuffle"]:
                self.camfiles = self.camfiles.shuffle()
        
        #get next chunk of files
        filenames = self.camfiles[self.ind: mfo + self.ind]
        
        #increment index to start with (modulo for circular effect)
        self.ind = (self.ind + kwargs["max_files_open"] ) % len(self.camfiles)
        
        return self.grab_data_chunk(filenames)
    
    def grab_data_chunk(self,filenames):
        """returns len(filenames)*4, 16, 768,1152 array"""
        
        #read in dataset
        dataset=MFDataset(filenames)
        
        #get every variable for every timestep across each file (var[i] is a len(filenames)*4, 768,1152 array )
        var = [dataset.variables[v][:] for v in self.kwargs["variables"]]

        #get every other time step (b/c only labelled in every other)
        labelled_vars = [v[::kwargs["time_step_stride"]] for v in var]

        #get some metadata
        n_tot_frames = sum([v.shape[0] for v in labelled_vars])
        xdim = labelled_vars[0].shape[1]
        ydim = labelled_vars[0].shape[2]
        time_steps = labelled_vars[0].shape[0]
        nvar = len(var)
        
        #interleave each variable together
        #tmp after this should be len(filenames)*4*nvar,768,1152
        #nvar = 16 usually
        tmp=np.empty((n_tot_frames,xdim,ydim ))
        for i in range(nvar):
            tmp[i::nvar,:] = labelled_vars[i]
        
        #now make tmp len(filenames)*4, 16, 768,1152 array
        tmp=tmp.reshape((time_steps, nvar, xdim, ydim))
        return tmp
        
        


              

    
    



if __name__ == "__main__":
    sys.path.insert(0,"/home/evan/hur-detect/scripts/")
    from configs import *
    kwargs = process_kwargs()
    for x,y in BBoxIterator(kwargs).iterate():
        print x.shape, y.shape

