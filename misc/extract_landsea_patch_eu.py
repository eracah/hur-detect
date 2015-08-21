"""
extract land sea mask image same size as atmospheric river imge size
Indicator for ocean is 1, for land is 0
"""
import os
import os.path as op
import numpy as np
import pandas as pd
import netCDF4 as cdf
import matplotlib 
matplotlib.use("agg")
from matplotlib.pyplot import *
import ipdb
import pickle


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
     this function is to cut the image patches 
     IN: d,  raw data field
     """
     imgs= []
     #for i in range(0, time): #note, range(a,b) results in (b-a) number instead of (b-a+1)
     xx=np.take(d,np.arange(latlow,lattop,-1),axis=0,mode='wrap')
     x=np.take(xx,np.arange(lonright,lonleft+1440), axis=1,mode='wrap') #wrap sround
     imgs+=[x]
     return imgs

"""
main code of data processing and image extracting
"""
#specify path to input/output data
camdir="/global/project/projectdirs/mantissa/climate/ERA-Interim"
outdir="/global/project/projectdirs/mantissa/climate/Yunjie/ar_path2/eu_new_samesize"

#define global grid of the land mask 
IJ=(721,1440) #lat, lon
I,J =IJ

# step size between grid points (in degrees)
dI = 180./(I-1)  #lat
dJ = 360./J    #lon

# grid locations
gI = np.arange(-90, 90+dI, dI)  #lat  note the lat is from 90 to -90, while generally it is from -90 to 90, need to reverse
gJ = np.arange(0, 360, dJ)   #lon

#define bounding box and grids location 
#I-bot=19.0N,I-top=56.0N,J-left=-180.0W,J-right=-120.0W
Ilow=25.0
Itop=60.0
Jleft=0
Jright=300.0  # convert to 0-360 scale

# define keys
keys= ["mask"]

#find the bounding box index (used for later slicing the patch )
latlow=I-nearestindex(Ilow,gI)
lattop=I-nearestindex(Itop,gI)
#latbound=(latlow,lattop)
lonleft=nearestindex(Jleft,gJ)
lonright=nearestindex(Jright,gJ)
#lonbound=(lonleft,lonright)

# access data file
fc=op.join(camdir,"era_interim_land_sea_mask.nc")
print ("current working file is ...")
print(fc)
dat=cdf.Dataset(fc,'r')
#print dat.data_model
print ("variables in the data file are ...")
#print dat.variables

#loop through time step (day) and  cut the patch
allimgs=dict([(k,[]) for k in keys])   #define it as dictionary 
for k in keys:
    dim=dat.variables[k].shape
    imgs=cut_box(abs(dat.variables[k][0]-1))
    allimgs[k]+=imgs

#save imags into pkl file
outfile=op.join(outdir,"landmask")
save_event(outfile+'_imgs_eu.pkl',allimgs)
       
        
