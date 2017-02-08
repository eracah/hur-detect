
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
import os
import sys
import time
import inspect
import copy
from util import get_camfiles, normalize,convert_nc_data_to_tensor, index_dict, vstack_dicts, dict_element_len
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
        if kwargs["im_dim"] == 3:
            assert kwargs["max_files_open"] >= kwargs['3d_time_steps_per_example'] /  (kwargs["time_steps_per_file"] / 
                                            kwargs["time_step_sample_frequency"]),"increase max_files open!"
       
    
    def iterate_chunks(self,batch_size=128):
        chunk_index = 0
        events_read =0 
        chunk = self.get_next_chunk()
        while events_read < self.num_events:
            if chunk_index + batch_size > dict_element_len(chunk):
                chunk = self.finish_out_chunk_and_get_as_many_more_as_needed(chunk, 
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
            
            ch = index_dict(chunk, excerpt)
            
            yield ch
            
    
    
    def finish_out_chunk_and_get_as_many_more_as_needed(self, chunk, ix, batch_size):
        chunk_len = dict_element_len(chunk)
        tmp = index_dict(chunk, slice(ix, chunk_len ))
        chunk = self.get_chunks_until_at_capacity(batch_size - dict_element_len(tmp))
        chunk = vstack_dicts(tmp,chunk)
        return chunk
        
    def get_chunks_until_at_capacity(self,batch_size):
        tmp = self.get_next_chunk()
        while  dict_element_len(tmp) < batch_size:
            chunk = self.get_next_chunk()
            tmp = vstack_dicts(tmp,chunk)
        return tmp

    def get_next_chunk(self):
        mfo = kwargs["max_files_open"]
        
        #if we are starting back up again shuffle everything
        if self.file_ind < mfo:
            if self.kwargs["shuffle"] and not self.kwargs["3D"]:
                self.camfiles = self.camfiles.shuffle()
        
        #get next chunk of files
        filenames = self.camfiles[self.file_ind: mfo + self.file_ind]
        
        #increment index to start with (modulo for circular effect)
        self.file_ind = (self.file_ind + kwargs["max_files_open"] ) % len(self.camfiles)
        
        return self._get_next_chunk(filenames)
        
        
    def _get_next_chunk(self,filenames):
        
        data_chunk = self.grab_data_chunk(filenames)
        
        label_chunk = self.grab_label_chunk(filenames)
        return {"data":data_chunk, "label": label_chunk}
    
    def grab_label_chunk(self,filenames):
        
        labels = make_yolo_masks_for_dataset(filenames[0],self.kwargs)
        if len(filenames) > 1:
            for f in filenames[1:]:
                label = make_yolo_masks_for_dataset(f,self.kwargs)
                labels = np.vstack((labels, label))
                
        if self.kwargs["im_dim"] == 3:
            time_steps = labels.shape[0]
            time_steps_per_example = self.kwargs["3d_time_steps_per_example"]
            labels = labels.reshape(time_steps / time_steps_per_example, 
                                       time_steps_per_example, 
                                       labels.shape[-3], 
                                       labels.shape[-2], 
                                       labels.shape[-1] )
            
            
        return labels
        
    
    def grab_data_chunk(self, filenames):
        """grabs input data (converts filenames to numpy tensors)
        returns len(filenames)*4, 16, 768,1152 array"""
        
        filenames = [join(self.kwargs["data_dir"],f) for f in filenames]


        dataset=MFDataset(filenames)
        
        tensor = convert_nc_data_to_tensor(dataset,self.kwargs)
 
        return tensor



if __name__ == "__main__":
    sys.path.insert(0,"/home/evan/hur-detect/scripts/")
    from configs import *
    kwargs = process_kwargs()
    kwargs["max_files_open"] = 4
    kwargs['num_val_days'] = 10
    t = time.time()
    for x in BBoxIterator(kwargs,mode="val").iterate_chunks(batch_size=1):
        print x['data'].shape, x['label'].shape
    print time.time() - t



# from matplotlib import pyplot as plt
# %matplotlib inline

# plt.imshow(x["data"][0][1])





