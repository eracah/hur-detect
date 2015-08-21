"""
Given bounding box, extract Atmospheric River image patches that making land fall over US-Canada west coast
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
define commonly used function
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
     this function cut interested image patches from raw data
     In: timea, time step that atmospheric river events happend
         timeb, time step that no atmospheric river events happend
         d, original TMQ field
     Out: cutted atmospheric river path and non atmospheric river patch
     """
     ar_imgs= []
     nonar_imgs=[]
     for i in timea:
         x=d[i,slice(latlow,lattop),slice(lonleft,lonright)]

         """
         U, V field has missing value at higher presusure level, convert masked array to normal array
         if the array itself is not masked, then return itself
         """
         x=np.ma.filled(x,np.random.uniform(0,0.01))
         ar_imgs+=[x]
     for j in timeb:
         y=d[j,slice(latlow,lattop),slice(lonleft,lonright)]
         y=np.ma.filled(y,fill_value=0)
         nonar_imgs+=[y]
     return [ar_imgs, nonar_imgs]



"""
main code of data processing and image patch extracting
"""
#specify the directory of the cam5 data, land mask data file and output directoty, hard coded
camdir="/global/project/projectdirs/m1248/suren_data/climate/cam5_tmq" #TMQ directory
outdir="/global/project/projectdirs/mantissa/climate/Yunjie/ar_patch2/us_new_samesize/"  #output directory 
landdir="/global/project/projectdirs/mantissa/climate/Yunjie"        # landmask directory

#define global grid of CAM5.0 output
IJ=(768,1152) #lat, lon
I,J =IJ

#step size between grid points (in degrees)
dI = 180./(I-1)  #lat
dJ = 360./J    #lon

# grid locations (in lat lon degrees)
gI = np.arange(-90, 90 + dI, dI)  #lat  
gJ = np.arange(0, 360, dJ)   #lon

#define bounding box
# I-bot=19.0N,I-top=56.0N,J-left=-180.0W,J-right=-120.0W (will create 158x224 image)
Ilow=20.0  
Itop=54.75 #will create 148x224 image
Jleft=180.0
Jright=250.0  # convert to 0-360 scale

#define keys, TMQ is required, I defined "date" because I am interested in when the atmospheric river happens
keys= ["TMQ","date"]#,"mask"]


#find the bounding box index (used for later slicing the patch )
latlow=nearestindex(Ilow,gI)
lattop=nearestindex(Itop,gI)
#latbound=(latlow,lattop)
lonleft=nearestindex(Jleft,gJ)
lonright=nearestindex(Jright,gJ)
#lonbound=(lonleft,lonright)


#read in atmospheric river date file
dfile=op.join(outdir,"AR_label_Jun8.txt")
with open(dfile) as df:
     dates=df.readlines()
newdates=[i[:-1] for i in dates]     
print("total numner of AR events are %d" %len(newdates))
#read in the non atmospheric date file
ddfile=op.join(outdir,"NonAR_label_Jun8.txt")
with open(ddfile) as ddf:
     dates1=ddf.readlines()
newdates1=[i[:-1] for i in dates1]   
print("total number of Non AR events are %d" %len(newdates1))

#read in the raw data file
for year in range(1979,2006):
    for month in range(1,13):
        """
        TODO: maybe use glob will make code more concise? 
        """
        tmqf=op.join(camdir,"TMQ_cam5_1_amip_run2.cam2.h1.%04d-%02d.nc" %(year,month))
        #uf=op.join(camdir1,"xU_cam5_1_amip_run2.cam2.h1.%04d-%02d.nc" %(year,month))
        #vf=op.join(camdir1,"xV_cam5_1_amip_run2.cam2.h1.%04d-%02d.nc" %(year,month))    
        #pslf=op.join(camdir1,"PSL_cam5_1_amip_run2.cam2.h1.%04d-%02d.nc" %(year, month))
        print ("current working on file ...")
        print(tmqf,"\n")
        tmqdat=cdf.Dataset(tmqf,'r')
        #udat=cdf.Dataset(uf,"r")
        #vdat=cdf.Dataset(vf,"r")
        #psldat=cdf.Dataset(pslf,'r') 

        #loop through time step (day) and  cut the patch
        ar_allimgs=dict([(k,[]) for k in keys])   # dictionary
        nonar_allimgs=dict([(k,[]) for k in keys])   

        dim=tmqdat.variables["TMQ"].shape  # all variables should have same time dimenion
        timestep=dim[0]    #the number of days withint a month 

        #get the atmospheric/non atmospheric river dates 
        AR_day=list()
        Non_AR_day=list()
        AR_date=list()
        Non_AR_date=list() 
        for day in range(0,timestep):
            dd="%04d-%02d-%02d" %(year,month,day+1)
            for ddd in newdates:
                 if (dd==ddd):
                    AR_day.append(day)
                    AR_date.append(dd)
            for ddd in newdates1:
                 if (dd==ddd):
                    Non_AR_day.append(day)
                    Non_AR_date.append(dd)
        print "days atmospheric river happend..."
        print AR_day
     
        print "days that no atmospheric river happend..."
        print Non_AR_day     

        # loop through keys to cut the image patch
        for k in keys:
            if k=='TMQ':
               [ar_imgs, nonar_imgs]=cut_box(tmqdat.variables[k],AR_day,Non_AR_day)
               ar_allimgs[k]+=ar_imgs
               nonar_allimgs[k]+=nonar_imgs
            if k=='date':
               ar_allimgs[k]+=AR_date
               nonar_allimgs[k]+=Non_AR_date
               #print tmqdat.variables[k].shape
            """
            # other variable keys might be interesting later
            if k=='PSL':
               [ar_imgs, nonar_imgs]=cut_box(psldat.variables[k],AR_day,Non_AR_day)
               ar_allimgs[k]+=ar_imgs
               nonar_allimgs[k]+=nonar_imgs
            if k=='U':
               #pick 850hPa wind level, same for V
               [ar_imgs, nonar_imgs]=cut_box(udat.variables[k][:,0,:,:],AR_day,Non_AR_day)
               ar_allimgs[k]+=ar_imgs
               nonar_allimgs[k]+=nonar_imgs   
            if k=='V':
               [ar_imgs, nonar_imgs]=cut_box(vdat.variables[k][:,0,:,:],AR_day,Non_AR_day)
               ar_allimgs[k]+=ar_imgs
               nonar_allimgs[k]+=nonar_imgs                        
            """
    
        #save all AR and Non_AR iamges into python serial file
        outfile=op.join(outdir,"atmosphericriver-%04d-%02d" %(year,month))
        if (ar_allimgs[k] != []): # if there is ar events, write them to file
            save_event(outfile+'_us_ar_imgs.pkl',ar_allimgs)
        if (nonar_allimgs[k]!=[]):
            save_event(outfile+'_us_nonar_imgs.pkl',nonar_allimgs)
