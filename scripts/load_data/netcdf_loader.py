
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
import copy
from util import get_camfiles, normalize,convert_nc_data_to_tensor
from ground_truth_maker import make_yolo_masks_for_dataset



class BBoxIterator(object):
    def __init__(self,kwargs,mode="tr"):
        self.kwargs = kwargs
        self.mode = mode
        #frame = inspect.currentframe()
        self.seed = 7
        self.camfiles = get_camfiles(self.kwargs["data_dir"], self.kwargs[mode+ "_years"])[:self.kwargs["num_" + mode + "_days"]]
        self.file_ind = 0
        self.chunk_ind =0 
        self.data_chunk=[]
        self.num_events = len(self.camfiles) * 4
        self.events_open = 4 * self.kwargs["max_files_open"]
       
    
    def iterate_chunks(self,batch_size=128):
        chunk_index = 0
        events_read =0 
        data_chunk = self.get_next_chunk()
        while events_read < self.num_events:
            if chunk_index + batch_size > len(data_chunk):
                data_chunk = self.finish_out_chunk_and_get_as_many_more_as_needed(data_chunk, 
                                                                                  chunk_index, 
                                                                                  batch_size)
                #back to 0 b/c we have a brand new chunk
                chunk_index = 0
            
            if events_read + batch_size > self.num_events:
                sm_batch_size = self.num_events - events_read
                excerpt = slice(chunk_index, chunk_index + sm_batch_size)
            
            else:

                #otherwise just read an excerpt from the current chunk
                excerpt = slice(chunk_index, chunk_index + batch_size)
            
            
            chunk_index += batch_size
            events_read += batch_size
            
            yield data_chunk[excerpt]
            
     
    def finish_out_chunk_and_get_as_many_more_as_needed(self,data_chunk,ix, batch_size):
        tmp = data_chunk[ix:]
        data_chunk = self.get_chunks_until_at_capacity(batch_size - tmp.shape[0])
        data_chunk = np.vstack((tmp,data_chunk))
        return data_chunk
        
    def get_chunks_until_at_capacity(self,batch_size):
        tmp = self.get_next_chunk()
        while tmp.shape[0] < batch_size:
            data_chunk = self.get_next_chunk()
            tmp = np.vstack((tmp,data_chunk))
        return tmp

    def get_next_chunk(self):
        mfo = kwargs["max_files_open"]
        
        #if we are starting back up again shuffle everything
        if self.file_ind < mfo:
            if self.kwargs["shuffle"]:
                self.camfiles = self.camfiles.shuffle()
        
        #get next chunk of files
        filenames = self.camfiles[self.file_ind: mfo + self.file_ind]
        
        #increment index to start with (modulo for circular effect)
        self.file_ind = (self.file_ind + kwargs["max_files_open"] ) % len(self.camfiles)
        
        return self._get_next_chunk(filenames)
        
        
    def _get_next_chunk(self,filenames):
        
        data_chunk = self.grab_data_chunk(filenames)
        
        #self.label_chunk = self.grab_label_chunk(filenames)
        return data_chunk
        
    
    def grab_data_chunk(self, filenames):
        """grabs input data (converts filenames to numpy tensors)
        returns len(filenames)*4, 16, 768,1152 array"""
        
        filenames = [join(self.kwargs["data_dir"],f) for f in filenames]


        dataset=MFDataset(filenames)
        
        tensor = convert_nc_data_to_tensor(dataset,self.kwargs)
 
        return tensor
        #if 3D -> convert to 3D



if __name__ == "__main__":
    sys.path.insert(0,"/home/evan/hur-detect/scripts/")
    from configs import *
    kwargs = process_kwargs()
    kwargs["max_files_open"] = 1
    kwargs['num_val_days'] = 3
    t = time.time()
    for x in BBoxIterator(kwargs,mode="val").iterate_chunks(batch_size=5):
        print x.shape
    print time.time() - t





