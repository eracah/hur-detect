#to merge several h5 file into a single one

import os
import numpy
import h5py
import ipdb



#data
EU="/global/project/projectdirs/mantissa/climate/Yunjie/ar_patch2/eu_new_samesize/atmosphericriver_eu+TMQ_Sep3.h5"
US="/global/project/projectdirs/mantissa/climate/Yunjie/ar_patch2/us_new_samesize/atmosphericriver_us+TMQ_Sep3.h5"


# read int the data and get the variables
Datset_us=h5py.File(US,"r")
Datset_eu=h5py.File(EU,"r")

big_AR=numpy.vstack((numpy.array(Datset_us['AR']),numpy.array(Datset_eu['AR'])))
big_NonAR=numpy.vstack((numpy.array(Datset_us['Non_AR']),numpy.array(Datset_eu['Non_AR'])))
big_AR_date=numpy.append(numpy.array(Datset_us['AR_date']),numpy.array(Datset_eu['AR_date']))
big_NonAR_date=numpy.append(numpy.array(Datset_us['NonAR_date']),numpy.array(Datset_eu['NonAR_date']))

#open new h5v file and create data set
f=h5py.File("merge.h5","w")
one=f.create_dataset('AR',big_AR.shape,dtype=numpy.float32)
zero=f.create_dataset('Non_AR',big_NonAR.shape,dtype=numpy.float32)
one_date=f.create_dataset('AR_date',big_AR_date.shape,dtype=numpy.dtype('S15'))
zero_date=f.create_dataset('NonAR_date',big_NonAR_date.shape,dtype=numpy.dtype('S15'))


#randomly shuffle the data
rr=range(one.shape[0])
numpy.random.shuffle(rr)
ipdb.set_trace()
big_AR=big_AR[rr]
big_AR_date=big_AR_date[rr]

rrp=range(zero.shape[0])
numpy.random.shuffle(rrp)
big_NonAR=big_NonAR[rrp]
big_NonAR_date=big_NonAR_date[rrp]


# assign data to dataset
one[:,:,:,:]=big_AR
zero[:,:,:,:]=big_NonAR
one_date[:]=big_AR_date
zero_date[:]=big_NonAR_date

#colse file
f.close()

