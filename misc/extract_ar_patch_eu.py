"""
Given bounding box, extract atmospheric river image pathes that makes land fall over europe west coast 
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

def  cut_box(d, timea,timeb):
     """
     Adapted from Sang's code 
     This function is different from the function used in the code of extracting atmospheric river image patches
     that makes landfall over US west coast, since we might need to across the boundary of the field
     In: timea, time step that atmospheric river events happend
         timeb, time step that no atmospheric river events happend
         d, original TMQ field
     Out: cutted atmospheric river path and non atmospheric river patch
     """
     # also need to pay attention that the latitude is reversed in ERA reanalysis data
     ar_imgs= []
     nonar_imgs=[]
     for i in timea:
         xx=np.take(d[i],np.arange(latlow,lattop,-1),axis=0,mode='wrap')  # the lat does not really need to wrap, because it does not cross boundary
         x=np.take(xx,np.arange(lonright,lonleft+1440),axis=1,mode='wrap') #here to wrap it across 0 longitude (boundary), where index in 1440
         ar_imgs+=[x]
     for j in timeb:
         yy=np.take(d[j],np.arange(latlow,lattop,-1),axis=0,mode='wrap')
         y=np.take(yy,np.arange(lonright,lonleft+1440),axis=1,mode='wrap')
         nonar_imgs+=[y]
     return [ar_imgs, nonar_imgs]


"""
main code of data processing and image patch extracting
"""
#specify the directory of the ERA-interim data
camdir="/global/project/projectdirs/mantissa/climate/ERA-Interim"
outdir="/global/project/projectdirs/mantissa/climate/Yunjie/ar_patch2/eu_new_samesize/"
nddir="/global/project/projectdirs/mantissa/climate/Yunjie"        # landmask directory

#define global grid of ERA_interim output
IJ=(721,1440) #lat, lon
I,J =IJ

#step size between grid points (in degrees)
dI = 180./(I-1)  #lat
dJ = 360./J    #lon

# grid locations
gI = np.arange(-90, 90+dI, dI)  #lat  Note: in ERA-interim the lat is just opposite, from 90 to -90
gJ = np.arange(0, 360, dJ)   #lon

#define bounding box
#I-bot=25.0N,I-top=60.0N,J-left=300.0W,J-right=0.0E (will create 140x240 image)
Ilow=24.0
Itop=61.0  #thi swill create image size(148x224)
Jleft=0.0
Jright=304.0  # convert to 0-360 scale, note later there value will ber wrapped 

# define keys
keys= ["TMQ","date"]#,"mask" 

#find the bounding box index (used for later slicing the patch )
latlow=I-nearestindex(Ilow,gI) #since the lat is reversed in ERA reanalysis, we reverse it back
lattop=I-nearestindex(Itop,gI)
#latbound=(latlow,lattop)
lonleft=nearestindex(Jleft,gJ)
lonright=nearestindex(Jright,gJ)
#lonbound=(lonleft,lonright)

#read in the AR dates
dfile=op.join(outdir,"AR_label_Jun29.txt")
with open(dfile) as df:
     dates=df.readlines()
print("total number of AR events are %d" %len(dates))
newdates=[float(i) for i in dates]   #convert string to float

#read in the Non AR dates
ddfile=op.join(outdir,"Non_AR_label_Jun29.txt")
with open(ddfile) as ddf:
     dates1=ddf.readlines()
print("total number of Non AR events are %d" %len(dates1))
newdates1=[float(i) for i in dates1]   #convert string to float


#read in the raw data file
for year in range(1979,2012):
    for month in range(1,13):
        """
        TODO: maybe use glob will make the code more concise???
        """
        fc=op.join(camdir,"era_interim_TMQ_%04d-%02d.nc" %(year,month))
        print ("current working file is ...")
        print(fc,"\n")
        dat=cdf.Dataset(fc,'r')
        #print ("variables in the data file are ...")
        #print dat.variables

        #loop through time step (day) and  cut the patch
        ar_allimgs=dict([(k,[]) for k in keys])   #define it as dictionary 
        nonar_allimgs=dict([(k,[]) for k in keys])   #define it as dictionary         
        dim=dat.variables['TMQ'].shape
        timestep=dim[0]
        AR_day=list()
        Non_AR_day=list()
        AR_date=list()
        Non_AR_date=list()
        for day in range(0,timestep):
                dd=float(dat.variables['time'][day])
                for ddd in newdates:
                     if (dd==ddd):      
                       AR_day.append(day)
                       AR_date.append(str(dd))
                for ddd in newdates1:
                     if (dd==ddd):
                       Non_AR_day.append(day)
                       Non_AR_date.append(str(dd))   
        print ("days that atmospheric river happend...")              
        print (AR_day)     #get ar day

        print("days that no atmospheric river happend...")
        print Non_AR_day
        
        #loop through keys to cut the iamge patch
        for k in keys:
            if k=='TMQ':
               #the varaible value in ERA reanalysis is scaled, but python automatically scale them back to its true value by default
               [ar_imgs, nonar_imgs]=cut_box(dat.variables[k],AR_day,Non_AR_day)
               ar_allimgs[k]+=ar_imgs
               nonar_allimgs[k]+=nonar_imgs
            if k=='date':
               ar_allimgs[k]+=AR_date 
               nonar_allimgs[k]+=Non_AR_date

        #save all atmospheric river and non atmospheric river image patches to python pickle file
        outfile=op.join(outdir,"atmosphericriver-%04d-%02d" %(year,month))
        if (ar_allimgs[k] != []): # if there is ar events, save it
            save_event(outfile+'_eu_ar_imgs.pkl',ar_allimgs)
        if (nonar_allimgs[k]!=[]):
            save_event(outfile+'_eu_nonar_imgs.pkl',nonar_allimgs)        
