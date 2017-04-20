


import sys
import os
import numpy as np
import random
# from configs import configs as climate_configs
import collections
from util import get_files_from_list, get_files_from_dir
#put home on path
# from dotpy_src.load_data.configs import configs as configs
#from images.netcdf_data_structure import ClimateNC
import h5py



def get_data(type_, data_file, file_format="h5"):
    if file_format == "nc":
        raise NotImplementedError
    else:
        return _get_data_h5(type_, data_file)



def _get_data_h5(type_, data_file):
    h5f = h5py.File(data_file)
    images = h5f["images"]
    labels = convert_box_tensor_to_box_lists(h5f["boxes"])
    return images, labels

    
    
    



def convert_box_tensor_to_box_lists(box_tensor):
    '''box tensor is a num_examples, max_num_boxes, 5 tensor
       for n boxes there are max_num_boxes - n rows filled (the rest of the rows are zero, so we 
       need to filter out the "zero rows" '''
    
    return [list(filter_out_negative_one_rows(box_array)) for box_array in box_tensor]
    



def filter_out_negative_one_rows(array, axis=1):
    '''takes n x 5 tensor'''
    return array[np.any(array > -1,axis=axis)]



if __name__ == "__main__":
    test = np.concatenate((np.ones((5,5)),np.zeros((10,5))))
    test= np.expand_dims(test,axis=0)
    ntest = np.concatenate((test,test,test))
    print convert_box_tensor_to_box_lists(ntest)





