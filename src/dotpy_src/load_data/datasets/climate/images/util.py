


import re
import numpy as np
import datetime as dt
import os
import sys
import numpy as np
import time
import h5py
import os
from os.path import join
import pandas as pd



def get_timestamp(filename):
    #print filename
    rpyear = re.compile(r"(\.h2\.)(.*?)(-)")
    rpdaymonth = re.compile(r"(-)(.*?)(\d{5}\.)")
    year=int(rpyear.search(filename).groups()[1])
    tmp=rpdaymonth.search(filename).groups()[1].split('-')
    month=int(tmp[0])
    day=int(tmp[1])
    return dt.date(year,month,day)



def normalize(arr,min_=None, max_=None, axis=(0,2,3)):
        if min_ is None or max_ is None:
            min_ = arr.min(axis=(0,2,3), keepdims=True)

            max_ = arr.max(axis=(0,2,3), keepdims=True)

        midrange = (max_ + min_) / 2.

        range_ = (max_ - min_) / 2.
        #print range_
        arr -= midrange
        arr /= (range_)
        return arr, min_, max_   



def get_camfiles(data_dir, years, with_dir=False):
    lsdir=os.listdir(data_dir)
    rpfile = re.compile(r"^cam5_.*\.nc$")
    camfiles = [f for f in lsdir if rpfile.match(f)]
    camfiles = [c for c in camfiles if get_timestamp(c).year in years]
    camfiles.sort()
    if with_dir:
        camfiles = [join(data_dir, camfile) for camfile in camfiles]
    return camfiles



def interleave_variables(labelled_vars, time_steps_per_example, label=False):
    #get some metadata
    n_tot_frames = sum([v.shape[0] for v in labelled_vars])
    xdim = labelled_vars[0].shape[1]
    ydim = labelled_vars[0].shape[2]
    time_steps = labelled_vars[0].shape[0]
    nvar = len(labelled_vars)
    
    def interleave_2d(labelled_vars):
        #interleave each variable together
        #tmp after this should be len(filenames)*4*nvar,768,1152
        #nvar = 16 usually
        tmp=np.empty((n_tot_frames,xdim,ydim ))
        for i in range(nvar):
            tmp[i::nvar,:] = labelled_vars[i]

        #now make tmp len(filenames)*4, 16, 768,1152 array
        tmp=tmp.reshape((time_steps, nvar, xdim, ydim))
        return tmp
        
    def interleave_3d(labelled_vars, time_steps_per_example):
        #interleaves each example in 3D fashion ,so takes k frames from each variable and
        #concatenates them where k is time_steps_per_example
        
        num_ex = time_steps / time_steps_per_example
        
        tmp=np.empty((num_ex, nvar, time_steps_per_example, xdim, ydim))
        for ex_ind in range(num_ex):
            for var_ind in range(nvar):
                tmp[ex_ind, var_ind,:time_steps_per_example,:] = labelled_vars[var_ind][ex_ind:ex_ind 
                                                                                        + time_steps_per_example]

        return tmp
    
    if time_steps_per_example > 1:
        tens= interleave_3d(labelled_vars, time_steps_per_example)
        if not label:
            tens, _, _ = normalize(tens, axis=(0,2,3,4))
    else:
        tens = interleave_2d(labelled_vars)
        if not label:
            tens, _, _ = normalize(tens)
        
    return tens
        
        

   
        



def get_boxes_for_nc_file(filepath,
                          path_to_csv_file="/home/evan/data/climate/climo/csv_labels/labels.csv",
                          time_step_freq=2, 
                          ims_per_file = 8):

    ims_per_file /= time_step_freq
    boxes = [ims_per_file*[]]
    filename = os.path.basename(filepath)
    
    coord_keys = ["xmin", "xmax", "ymin", "ymax"]
    cls_key = "category"
    
    labeldf = pd.read_csv(path_to_csv_file)
    
    
    ts = get_timestamp(filename)

    filedf = labeldf.ix[ (labeldf.month==ts.month) & (labeldf.day==ts.day) & (labeldf.year==ts.year) ].copy()


    time_steps = [ int(num) for num in list(filedf["time_step"])]

    box_coords = filedf[["xmin", "xmax", "ymin", "ymax", "category"]].values

    boxes = [[] for i in range(ims_per_file)]
    time_step_boxes_pairs = zip(time_steps, box_coords)
    for time_step, box in time_step_boxes_pairs:
        boxes[time_step / 2].append(list(box))
    
    return boxes


def convert_list_box_lists_to_np_array(list_of_box_lists, max_rows=15):
    num_ex = len(list_of_box_lists)
    
    #the first box of the first list
    sample_item = list_of_box_lists[0][0]
    

    arr = -1*np.ones((num_ex,max_rows, len(sample_item)))
    for box_list_ind,box_list in enumerate(list_of_box_lists):
        num_used_rows = len(box_list)
        if len(box_list) > 0:
            rows = np.vstack(tuple(box_list))
            arr[box_list_ind,:num_used_rows] = rows
    return arr
        



def convert_nc_data_to_tensor(dataset,variables, is_label,
                                           time_step_sample_freq, time_steps_per_example):
        #get every variable for every timestep across each file (var[i] is a len(filenames)*4, 768,1152 array )
        var = [dataset.variables[v][:] for v in variables]

        #get every other time step (b/c only labelled in every other)
        labelled_vars = [v[::time_step_sample_freq] for v in var]
        
        tensor = interleave_variables(labelled_vars,time_steps_per_example, label=is_label)
        return tensor

