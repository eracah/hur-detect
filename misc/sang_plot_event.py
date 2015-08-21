# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import os.path as op
import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
import netCDF4 as cdf

import matplotlib
matplotlib.use('agg')
from matplotlib.pyplot import *

# <codecell>

TCdir = op.join(os.environ['HOME'], 'Data', 'climate', 'TC_analysis')
keys = [u'PSL', u'T200', u'T500', u'TMQ', u'TS',
        u'U850', u'UBOT', u'V850', u'VBOT']

# number of grid points
# I: hours
# J: latitude
# K: longitude
IJK = (8, 768, 1152)
I, J, K = IJK

# step size between grid points (in degrees)
dJ = 180./(J-1)
dK = 360./K

# grid locations
gI = np.arange(0, 23, 3)
gJ = np.arange(-90, 90 + dJ, dJ)
gK = np.arange(0, 360, dK)

# <codecell>

def getfnames(topdir, searchstr):
    from fnmatch import filter
    from os.path import join
    from os import walk

    matches = []
    for root, dirnames, filenames in walk(topdir):
      for filename in filter(filenames, searchstr):
          matches.append(join(root, filename))

    return matches

def load_events(f):
    from pickle import load
    with open(f, 'rb') as fp:
        ev = load(fp)        
    return ev

# <codecell>

#prefix = sys.argv[1]
prefix = 'hurricanes'
ifile = getfnames('/home/syoh/Projects/climate/', prefix + '*imgs.pkl')
efile = getfnames('/home/syoh/Projects/climate/', prefix + '*list.pkl')
ifile.sort()
efile.sort()

# <codecell>

for imgf, evtf in zip(ifile, efile):
    img = load_events(imgf)
    evt = load_events(evtf)

    keys = img.keys()
    keys.sort()

    print imgf, evtf
    print keys

    #for i_ymd, ymd in evt.groupby('')
    fig, axes = subplots(nrows=len(evt), ncols=len(keys), figsize=(5*len(keys), 5*len(evt)))
    subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.01, hspace=0.01)

    for i, row in enumerate(axes):
        for k, ax in zip(keys, row): 
            ax.imshow(img[k][i], interpolation='none')
            

    root, ext = op.splitext(imgf)
    savefig(root + '.png')
