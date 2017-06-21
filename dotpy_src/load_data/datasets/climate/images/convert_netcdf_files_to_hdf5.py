


import netCDF4 as nc

import h5py

import sys
import os
from os import listdir
from os.path import join
import pandas as pd
import re

from util import get_camfiles, convert_nc_data_to_tensor, convert_list_box_lists_to_np_array, get_boxes_for_nc_file
import time



variables=['PRECT','PS','PSL','QREFHT','T200','T500','TMQ','TREFHT',
                                  'TS','U850','UBOT','V850','VBOT','Z1000','Z200','ZBOT']



ims_per_file = 8
time_step_sample_freq=2
max_rows = 15
box_dim =5 #4 coords plus class
ims_per_file /= time_step_sample_freq



def get_h5f(prefix, years, dest_path):
    h5filename = prefix+"climo_" + "_".join([str(y) for y in years]) + ".h5"
    h5filepath = join(dest_path, h5filename)
    h5f = h5py.File(h5filepath, "w")
    return h5f



def get_ds(h5f, camfiles, num_ims):
    all_ims = nc.MFDataset(camfiles)
    num_ex = all_ims[variables[0]].shape[0] / time_step_sample_freq if num_ims == -1 else num_ims
    xdim, ydim = all_ims[variables[0]].shape[1], all_ims[variables[0]].shape[2]

    all_ims.close()
    im_ds = h5f.create_dataset(name="images",shape=(num_ex,len(variables), xdim, ydim), dtype="f4",compression="gzip" )
    box_ds = h5f.create_dataset(name="boxes", shape=(num_ex,max_rows,box_dim ), dtype="i4",compression="gzip")
    

    return im_ds, box_ds

    



def get_im_box_arrays(camfile):
    ncd = nc.MFDataset(camfile)
    im_arr = convert_nc_data_to_tensor(ncd,variables=variables,is_label=False, 
                                         time_step_sample_freq=time_step_sample_freq, time_steps_per_example=1)
    box_list_arr = convert_list_box_lists_to_np_array(get_boxes_for_nc_file(camfile), boxdim=5)
    return  im_arr, box_list_arr



def copy_arrays_to_hdf5(im_arr, box_list_arr, add_to_ds, ind, num_ex):
    if ind + ims_per_file <= num_ex:
        add_to_ds(im_arr, box_list_arr, slice(ind, ind+ims_per_file))
    else:
        rest = num_ex - ind
        add_to_ds(im_arr[:rest], box_list_arr[:rest], slice(ind, num_ex))
    



def get_ds_func(im_ds, box_ds):
    def add_to_ds(im_arr, box_arr, slice_):
        im_ds[slice_] = im_arr
        box_ds[slice_] = box_arr
        print slice_.start, slice_.stop, im_arr.shape, box_arr.shape
    return add_to_ds



def convert_nc_to_h5(years=[1980], base_path="/home/evan/data/climate/climo/images/", dest_path="/home/evan/data/climate/",num_ims=-1, prefix=""):
    h5f = get_h5f(prefix, years, dest_path)
    
    camfiles = get_camfiles(data_dir=base_path, with_dir=True,years=years, ims_per_file=ims_per_file, num_ims=num_ims)
    im_ds, box_ds = get_ds(h5f, camfiles, num_ims)

    add_to_ds = get_ds_func(im_ds, box_ds)
    
    num_ex = im_ds.shape[0]
    
    ind = 0
    for camfile in camfiles:
        if ind >= num_ex:
            break
        else:
            im_arr, box_list_arr = get_im_box_arrays(camfile)
            copy_arrays_to_hdf5(im_arr, box_list_arr, add_to_ds, ind, num_ex)

        ind += ims_per_file



if __name__ == "__main__":
    convert_nc_to_h5(prefix="sample_", num_ims=32)








