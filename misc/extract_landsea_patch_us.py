"""
To extract land sea mask images the same size as the atmospheric river image patch
Indicator for land is 0, indicator for ocean is 1.
"""
import os
import os.path as op
import numpy as np
import pandas as pd
import netCDF4 as cdf
import matplotlib 
matplotlib.use("agg")
from matplotlib.pyplot import *
import pickle
import ipdb


"""
define commonly used functions
"""
def nearestindex(x,v):
    """
    Adapted from Sang's code. 
    this function find nearest index for x in array v
    In: v, array like
        x, given value
    Out: r, the nearest index of x in v 
    """
    r=np.searchsorted(v,x,"right")-1
    return r if np.linalg.norm(v[r]-x) <= np.linalg.norm(v[r+1]-x) else r+1

def save_event(f,ev):
    """
    Adapted from Sang's code
    this function save events ev to file f
    In: f, file name
        ev, events needs to be saved
    """
    with open(f, "wb") as fid:
          pickle.dump(ev,fid)

def  cut_box(d):
     """
     Adapted from Sang's code
     this function cut interested image patch from raw data
     IN: d, raw data field
     """
     imgs= []
     #for i in range(0, time): #note, range(a,b) results in (b-a) number instead of (b-a+1)
     x=d[0,0,slice(latlow,lattop),slice(lonleft,lonright)]
     imgs+=[x]
     return imgs

"""
main code of data processing and image extraction
"""
#specify in/out data path
camdir="/global/project/projectdirs/mantissa/climate/Yunjie"
outdir="/global/project/projectdirs/mantissa/climate/Yunjie/ar_patch2/us_new_samesize/"

#define global grid of the land mask 
IJ=(768,1152) #lat, lon
I,J =IJ

# step size between grid points (in degrees)
dI = 180./I  #lat
dJ = 360./J    #lon

# grid locations
gI = np.arange(-90, 90, dI)  #lat  
gJ = np.arange(0, 360, dJ)   #lon

#define bounding box and grids location  
#I-bot=19.0N,I-top=56.0N,J-left=-180.0W,J-right=-120.0W

Ilow= 19.0
Itop=56.0
Jleft=180.0
Jright=250.0  # convert to 0-360 scale

# define keys
keys= ["mask"]

#find the bounding box index (used for later slicing the patch )
latlow=nearestindex(Ilow,gI)
lattop=nearestindex(Itop,gI)
#latbound=(latlow,lattop)
lonleft=nearestindex(Jleft,gJ)
lonright=nearestindex(Jright,gJ)
#lonbound=(lonleft,lonright)

# access data file
fc=op.join(camdir,"land-sea-mask-v2-regrid.nc")   # hard coded path and file name
print ("current working file is ...")
print(fc)
dat=cdf.Dataset(fc,'r')
#print dat.data_model
print ("variables in the data file are ...")
print dat.variables

allimgs=dict([(k,[]) for k in keys])   #define it as dictionary 
for k in keys:
    dim=dat.variables[k].shape
    imgs=cut_box(dat.variables[k])
    allimgs[k]+=imgs

#save imags into pkl file
outfile=op.join(outdir,"landmask")
save_event(outfile+'_imgs_us.pkl',allimgs)
