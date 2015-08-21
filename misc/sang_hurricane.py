# sang's code on extraction TC
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import os.path as op
import numpy as np #numpy is python  scientific computing module 
import pandas as pd #pandas is python data analysis module 
import sklearn.preprocessing as pp #sklearn is python machine learning module 
import netCDF4 as cdf # netCDF is python module to handle netCDF file

import matplotlib  #matplotlib is python 2d plotting library
matplotlib.use('agg') # backend plot raster PNG format 
#typically, this is called use "backend", "frontend" is the user facing code, 'backend' does all hard work behind the screen
from matplotlib.pyplot import * #matlab like plotting frame work, import everything

# <codecell>

TCdir = op.join(os.environ['HOME'], 'Data', 'climate', 'TC_analysis')
keys = [u'PSL', u'T200', u'T500', u'TMQ', u'TS',
        u'U850', u'UBOT', u'V850', u'VBOT']

# number of grid points
# I: hours
# J: latitude
# K: longitude
IJK = (8, 768, 1152) #tuple packing
I, J, K = IJK  #tuple unpacking

# step size between grid points (in degrees)
dJ = 180./(J-1)
dK = 360./K

# grid locations
gI = np.arange(0, 23, 3)
gJ = np.arange(-90, 90 + dJ, dJ)
gK = np.arange(0, 360, dK)

# <codecell>

def nearestindex(x, v):
    
    """
    find the index of point in v is closest to x
    """

    from numpy.linalg import norm
    from numpy import searchsorted

    r = searchsorted(v, x, 'right') - 1  # searchsorted: the index of the location in v that insert x will not change the order of v
    return r if norm(v[r]-x) <= norm(v[r+1]-x) else r+1

def hurricanetracks(f, hr, lat, lon):
    
    """
    read the hurricane tracks data from the 'visit' formatted text file
    """

    from numpy import arange, genfromtxt
    from pandas import DataFrame
    
    ctr = genfromtxt(f, delimiter=',', names=True, usecols=(0,1,2,3,4,5,6), 
                     dtype=('i','i','i','i','i','f','f'))  #load data from formatted file
    ctr = DataFrame(ctr)   # convert to data frame
    ctr['ihour'] = nearestindex(x=ctr['hour'], v=hr) ## matrix index 
    ctr['ilon'] = nearestindex(x=ctr['lon'], v=lon)  ## nearest matrix index
    ctr['ilat'] = nearestindex(x=ctr['lat'], v=lat)  ## nearest matrix index
    ctr['file'] = ['cam5_1_amip_run2.cam2.h2.%4d-%02d-%02d-00000.nc' % 
                   (one['year'], one['month'], one['day']) 
                   for i, one in ctr.iterrows()]
    
    return ctr

def save_events(f, ev):  # save to pickle file

    from pickle import dump

    with open(f, 'wb') as fp:
        dump(ev, fp)

# <codecell>

def comp_box(dims, info, patch_size=32):

    """
    from output of 'hurricanetracks' in info, compute the outer bounds of image patches of size patch_size
    allok output marks if each image patch is within bounds of the image
    """
    
    imax, jmax, kmax = dims
    s = patch_size/2
    
    jintv = [(a, b) for a, b in zip(info['ilat']-s, info['ilat']+s+1)]  #zip returns a list of tuples, where the i-th tuple contains i-th element from argumentsequence
    kintv = [(a, b) for a, b in zip(info['ilon']-s, info['ilon']+s+1)]

    jok = [all((0 <= a, b < jmax)) for a, b in jintv]  # all returen true if all elements of the iterables are true
    kok = [all((0 <= a, b < kmax)) for a, b in kintv]
    allok = [a and b for a, b in zip(jok, kok)]
    
    return [one for one in zip(info['ihour'], jintv, kintv, allok)]

def cut_box(d, bounds, dims=None):
    
    """
    get image patches from d
    """

    from numpy import take, arange, remainder
    
    imax, jmax, kmax = d.shape if dims is None else dims
    
    imgs = []
    for i_, j_, k_, ok_ in bounds:

        if ok_:
            # simple slicing
            x = d[i_, slice(*j_), slice(*k_)]   # not sure how slice works here ???/
        else:
            # need to wrap around
            x = take(d[i_], remainder(arange(*j_), jmax), axis=0, mode='wrap')
            x = take(x, remainder(arange(*k_), kmax), axis=1, mode='wrap')

        imgs += [x]

    return imgs

def image_squares(info, keys, prefix, patch_size=32):
    
    """
    coordinates reading image patches in info by grouping events by simulation data file
    """
    
    from os.path import join
    from pandas import concat
    
    evimg = dict([(k,[]) for k in keys])
    evlst = None
    for f, ev in info.groupby('file'):

        d = cdf.Dataset(join(prefix, f), 'r')
        evlst = concat((evlst, ev))
        dshape = d.variables['TMQ'].shape
        imbox = comp_box(dshape, ev, patch_size)
        for k in keys:
            imgs = cut_box(d.variables[k], imbox)
            evimg[k] += imgs

    return evlst, evimg

# <codecell>

years = range(1979, 2007)
fs = [op.join(TCdir, str(one), 'visit_output.txt') for one in years] # file names
ctr = pd.concat([hurricanetracks(f, gI, gJ, gK) for f in fs])        # all hurricane centers

# <codecell>

## subset images for testing. delete for full run
f5 = ['cam5_1_amip_run2.cam2.h2.1979-08-07-00000.nc',
      'cam5_1_amip_run2.cam2.h2.1979-09-04-00000.nc']
ctr = ctr[[one in f5 for one in ctr['file']]]

# <codecell>

# files can get big, so save files by year and month
for i_yrmo, yrmo in ctr.groupby(['year', 'month']):

    ev, img = image_squares(yrmo, keys, TCdir)

    if len(img[keys[0]]) != len(yrmo):
        print "tracked points (%d) and image patch (%d) counts differ" % (len(img[keys[0]]), len(yrmo))

    # f = "hurricanes-%4d" % i_yrmo
    f = "hurricanes-%4d-%02d" % i_yrmo
    save_events(f+'_imgs.pkl', img)
    yrmo.to_pickle(f+'_list.pkl')    
