
import matplotlib; matplotlib.use("agg")


import re
import numpy as np
import datetime as dt
import os



def index_dict(d,excerpt):
    '''indexes every array in the dict and returns new dict'''
    for k,v in d.iteritems():
        d[k] = v[excerpt]
    return d



def vstack_dicts(d1,d2):
    '''vstacks two dicts'''
    assert d1.keys() == d2.keys(), "can't combine two dicts with diff keys!"
    for k,v in d1.iteritems():
        d1[k] = np.vstack((v,d2[k]))
    
    return d1
    



def dict_element_len(d):
    '''gets the length of a random element in the dict'''
    return d[d.keys()[0]].shape[0]



def normalize(arr,min_=None, max_=None, axis=(0,2,3)):
        if min_ is None or max_ is None:
            min_ = arr.min(axis=(0,2,3), keepdims=True)

            max_ = arr.max(axis=(0,2,3), keepdims=True)

        midrange = (max_ + min_) / 2.

        range_ = (max_ - min_) / 2.
        
        arr -= midrange

        arr /= (range_)
        return arr, min_, max_   




def convert_bbox_minmax_to_cent_xywh(bboxes):
    #current bbox set up is xmin,ymin,xmax,ymax
    xmin, xmax,ymin,  ymax = [ bboxes[:,i] for i in range(4) ]
    
    w = xmax - xmin
    h = ymax - ymin

    x_c = xmin + w / 2.
    y_c = ymin + h / 2.
    
    
    bboxes[:,0] = x_c
    bboxes[:,1] = y_c
    bboxes[:,2] = w # w
    bboxes[:,3] = h #h
    return bboxes



def get_timestamp(filename):
    #print filename
    rpyear = re.compile(r"(\.h2\.)(.*?)(-)")
    rpdaymonth = re.compile(r"(-)(.*?)(\d{5}\.)")
    year=int(rpyear.search(filename).groups()[1])
    tmp=rpdaymonth.search(filename).groups()[1].split('-')
    month=int(tmp[0])
    day=int(tmp[1])
    return dt.date(year,month,day)



def get_camfiles(data_dir, years):
    lsdir=os.listdir(data_dir)
    rpfile = re.compile(r"^cam5_.*\.nc$")
    camfiles = [f for f in lsdir if rpfile.match(f)]
    camfiles = [c for c in camfiles if get_timestamp(c).year in years]
    camfiles.sort()
    return camfiles



def interleave_variables(labelled_vars, kwargs, dim=2):
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
    
    if kwargs["im_dim"] == 3:
        return interleave_3d(labelled_vars, kwargs["3d_time_steps_per_example"])
    else:
        return interleave_2d(labelled_vars)
        
        

   
        



def convert_nc_data_to_tensor(dataset,kwargs):
        #get every variable for every timestep across each file (var[i] is a len(filenames)*4, 768,1152 array )
        var = [dataset.variables[v][:] for v in kwargs["variables"]]

        #get every other time step (b/c only labelled in every other)
        labelled_vars = [v[::kwargs["time_step_sample_frequency"]] for v in var]
        
        tensor = interleave_variables(labelled_vars,kwargs)
        return tensor



if __name__ == "__main__":

    a ={"a":[2,3,4,5],"b":[3,4,5,6]}
    b = {"a":[1],"b":[3]}

    #c=vstack_dicts(a,b)
    d = index_dict(a, slice(1,3,1))





