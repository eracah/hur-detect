
# coding: utf-8

# In[1]:

import netCDF4 as nc

import h5py

import sys
import os
from os import listdir
from os.path import join
from nbfinder import NotebookFinder
sys.meta_path.append(NotebookFinder())
import pandas as pd
import re

from util import *


# In[2]:

variables=['PRECT','PS','PSL','QREFHT','T200','T500','TMQ','TREFHT',
                                  'TS','U850','UBOT','V850','VBOT','Z1000','Z200','ZBOT']



# In[33]:

def convert_nc_to_h5(year=1980, base_path="/home/evan/data/climate/climo/images/", dest_path="/home/evan/data/climate/",num_ims=-1, prefix=""):
    h5f = h5py.File(join(dest_path,prefix+"climo_" + str(year) + ".h5"), "w")
    
    ims_per_file = 8
    time_step_sample_freq=2
    ims_per_file /= time_step_sample_freq
    max_rows = 15
    box_dim =5 #4 coords plus class


    camfiles = get_camfiles(base_path,[year], with_dir=True )
    camfiles.sort()
    if num_ims != -1:
        num_camfiles = int(np.ceil(float(num_ims) / ims_per_file))
        camfiles = camfiles[:num_camfiles]
    
    all_ims = nc.MFDataset(camfiles)
                    
    num_ex = all_ims[variables[0]].shape[0] / time_step_sample_freq if num_ims == -1 else num_ims
    xdim, ydim = all_ims[variables[0]].shape[1], all_ims[variables[0]].shape[2]


    im_ds = h5f.create_dataset(name="images",shape=(num_ex,len(variables), xdim, ydim), dtype="f4",compression="gzip" )
    box_ds = h5f.create_dataset(name="boxes", shape=(num_ex,max_rows,box_dim ), dtype="i4",compression="gzip")

    ind = 0
    for cfile in camfiles:
        if ind >= num_ex:
            break
        ncd = nc.MFDataset(cfile)
        np_array = convert_nc_data_to_tensor(ncd,variables=variables,is_label=False,time_step_sample_freq=time_step_sample_freq,time_steps_per_example=1)
        
        box_list_arr = convert_list_box_lists_to_np_array(get_boxes_for_nc_file(cfile))
        if ind + ims_per_file <= num_ex:
            im_ds[ind:ind+ims_per_file] = np_array
            box_ds[ind:ind+ims_per_file] = box_list_arr
            
            print ind, ind+ims_per_file, np_array.shape, box_list_arr.shape
        else:
            rest = num_ex - ind
            im_ds[ind:num_ex] = np_array[:rest]
            box_ds[ind:num_ex] = box_list_arr[:rest]
            print ind, num_ex, np_array[:rest].shape, box_list_arr[:rest].shape
            
        
        
        ind += ims_per_file


# In[34]:

if __name__ == "__main__":
    convert_nc_to_h5()
    #h5f = h5py.File("/home/evan/data/climate/climo_1980.h5")

    # from matplotlib import pyplot as plt
    # %matplotlib inline

    # im = h5f["images"][280][6]
    # print im
    # plt.imshow(im,origin="lower")

    # im = h5f["images"][10][6]

    # plt.imshow(im,origin="lower")


# In[40]:

#! jupyter nbconvert convert_netcdf_files_to_hdf5.ipynb --to script


# In[ ]:



