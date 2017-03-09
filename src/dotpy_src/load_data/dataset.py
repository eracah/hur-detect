


import sys
import os

import netCDF4 as nc
from netCDF4 import MFDataset
from os import listdir, system
from os.path import isfile, join, isdir
import numpy as np
import imp
import itertools
import os
import sys
import time
import inspect
import copy
from util import get_camfiles,convert_nc_data_to_tensor
# from labels.yolo_maker import make_yolo_masks_for_dataset, make_multiple_yolo_masks
import random
from configs import configs
import collections



class DataSet(object):

    def __init__(self,images,labels):


        self._images = images
        self._labels = labels
        self._num_examples = self._images.total_examples
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def shuffle(self):
        pass

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            
            #shuffle files
            seed = np.random.randint(0,1000)
            self._images.shuffle(seed)
            self._labels.shuffle(seed)

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                seed = np.random.randint(0,1000)
                self._images.shuffle(seed)
                self._labels.shuffle(seed)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            
            
            return np.concatenate((images_rest_part, images_new_part), axis=0),                    np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
           
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            images = self._images[start:end]
            labels = self._labels[start:end]
            return images,labels



class ClimateImageOrLabel(object):
    def __init__(self,filepaths,shape,crop_indices=None, crop_stride=None, variables=["TMQ", "VBOT", "PSL"],
                 time_step_sample_freq=1, time_steps_per_example=1,time_steps_per_file=8):
        
        assert time_steps_per_example == 1, "3d not quite supported for labels"
        frame = inspect.currentframe()
        # set self.k = v for every k,v pair in __init__ except self of course
        self.set_constructor_args(frame)
        
        self.num_files = len(self.filepaths)
        self.examples_per_file = (time_steps_per_file / time_step_sample_freq) / time_steps_per_example
        self.total_examples = self.num_files * self.examples_per_file


    def set_constructor_args(self,frame):
        #set data members for object from constructor args
        _, _, _, params = inspect.getargvalues(frame)
        del params["frame"]
        for k,v in params.iteritems():
            setattr(self,k,v)
        
        
    
    def shuffle(self, seed):
        '''only shufflez files'''
        rng = np.random.RandomState(seed)
        random.shuffle(self.filepaths, random=rng.uniform)
        if self.crop_indices:
            rng = np.random.RandomState(seed)
            random.shuffle(self.crop_indices, random=rng.uniform)
            
        
        

    
    #overloading of bracket operators
    def __getitem__(self, slice_):
        slices = self.convert_slice_to_file_and_ex_inds(slice_)
        if slices is None:
            z_shape = tuple([0] + list(self.shape))
            return np.zeros(z_shape)
        file_slice = slices["file_slice"]
        ex_slice = slices["ex_slice"]
        filepaths = self.filepaths[file_slice]
        if self.crop_indices:
            crop_indices = self.crop_indices[file_slice]
        else:
            crop_indices = None
            
        tens = self.grab_data_chunk(filepaths, crop_indices)
#         lbls = make_multiple_yolo_masks(camfile_paths=filepaths,
#                                     labels_csv_file=self.labels_csv_file,
#                                     caffe_format=True)
        data = tens[ex_slice]
#         labels = lbls[ex_slice]
        return data
    
    def convert_slice_to_file_and_ex_inds(self, slice_):
        if isinstance(slice_, slice):
            start, stop, step = [getattr(slice_,k) for k in ["start", "stop", "step"]]
            assert step==1 or step is None, "step must be 1 or None"
        
        elif isinstance(slice_, int):
            start, stop = [slice_, slice_ + 1]
            
        slices =  self.get_file_and_ex_inds(start, stop)
        return slices
        
        
        
            
    def get_file_and_ex_inds(self, start, stop):
        if start == stop:
            return None
            
        #file start stop indices to index filenames
        file_start, file_stop = self.get_file_ind(start), self.get_file_ind(stop)
        
        # get some useful numbers
        tot_examples_desired = stop - start 
        
        #relative example indices after examples read in
        ex_start = self.get_relative_ex_ind(start)
        ex_stop = ex_start + tot_examples_desired

        file_slice = slice(file_start,file_stop)
        ex_slice = slice(ex_start,ex_stop)
        
        return {"file_slice":file_slice, "ex_slice": ex_slice}
    

    

    def get_file_ind(self,ex_ind):
        return ex_ind / self.examples_per_file

    def get_relative_ex_ind(self, ex_ind):
        return ex_ind % self.examples_per_file
        
        
        
    def grab_data_chunk(self, filepaths, crop_indices=None):
        """grabs input data (converts filepaths to np tensors)
        returns len(filepaths)*4, 16, 768,1152 array"""
        


        dsets=[]
        for filepath in filepaths:
            dataset=nc.Dataset(filepath)
            if "x_coord" in self.variables:
                is_label = True
            else:
                is_label = False
            tensor = convert_nc_data_to_tensor(dataset,
                                               self.variables, is_label,
                                               self.time_step_sample_freq,
                                               self.time_steps_per_example)
            xdim = tensor.shape[2]
            #hard code crop to 768
            tensor = tensor[:,:,:,:xdim]
            dsets.append(tensor)
        tensor = np.vstack(tuple(dsets))
            
        
 
        return tensor



def make_datasets(num_examples=-1, typ="tr"):
    #Datasets = collections.namedtuple('Datasets', ['tr', 'val', 'test'])#, "tr_unlabelled"])
    data_list_dir = configs["data_list_dir"]
    climate_data = make_dataset(data_list_dir, typ, num_examples)
    #tr, val, test = [make_dataset(data_list_dir,type_,num_examples=num_examples) for type_ in ['tr', 'val', 'test']]#, "tr_unlabelled"]]
    #climate_data = Datasets(tr=tr, val=val, test=test)#, tr_unlabelled=tr_unlabelled)
    return climate_data
    



def make_dataset(data_list_dir, type_, num_examples):
    
    im_files = get_files_from_list(data_list_dir, type_, "image")
    lbl_files = get_files_from_list(data_list_dir, type_, "label")
    if num_examples != -1:
        num_files = (configs["time_step_sample_frequency"] * num_examples) / configs["time_steps_per_file"]
        if num_files <= len(im_files):
            im_files = im_files[:num_files]
            lbl_files = lbl_files[:num_files]
    
    #crop_indices = get_files_from_list(data_list_dir, type_, "crop_indices")
    crop_indices=None
    
    
    #camfiles = get_camfiles(data_dir, configs[type_ + "_years"], with_dir=True)
    images = ClimateImageOrLabel(filepaths=im_files, shape=(16,768,768),
                                 crop_indices=crop_indices, 
                                 time_step_sample_freq=configs["time_step_sample_frequency"],
                                 variables=configs["image_variables"],
                                 time_steps_per_example=configs["time_steps_per_example"])
    
    
    labels = ClimateImageOrLabel(filepaths=lbl_files,shape=(6,24,24),
                                 crop_indices=crop_indices,
                                 variables=configs["label_variables"],
                                 time_step_sample_freq=configs["time_step_sample_frequency"],
                                 time_steps_per_example=configs["time_steps_per_example"])
    
    
    dataset = DataSet(images,labels)
    return dataset



def get_files_from_list(list_dir, type_, images_or_labels_or_crop_indices):
    with open(join(list_dir, type_ + "_" + images_or_labels_or_crop_indices + "_files.txt"), "r") as f:
        return [line.strip("\n") for line in f.readlines()]



if __name__ == "__main__":
    cl_data = make_datasets(num_tr_examples=22)

    im, lbl = cl_data.next_batch(batch_size=11)





