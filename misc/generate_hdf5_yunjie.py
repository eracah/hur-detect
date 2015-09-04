"""
write data into hdf5 format
"""
import logging
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import glob
import pickle  
import h5py
import ipdb
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ar(object):

    def __init__(self, **kwargs):
        """
        initialize from given argument
        """
        self.__dict__.update(kwargs)
        
    def hdf5(self,maxsize):
        """
        This function define routine of writing hdf5 file
        IN: maxsize, max block size of the data file
            shp, the shape of the data
            outfile, output hdf5 filename
        """
        # get a list of all interesting files
        pf = glob.glob(os.path.join(self.repo_path, 'atmosphericriver*_ar_imgs.pkl'))
        nf = glob.glob(os.path.join(self.repo_path, 'atmosphericriver*_nonar_imgs.pkl'))
        
        # get the land mask file if intersect is going to be calculated
        intersect=self.intersect
        if intersect:
           mf=glob.glob(os.path.join(self.repo_path,"landmask*.pkl"))
           logger.debug("land mask file  %s" %mf)
           with open(mf[0],"rb") as mfid:
                landsea=pickle.load(mfid)
           landsea=abs(np.asarray(landsea['mask'])-1) # convert indicator 1 for land
           logger.debug("read in and processed the land sea information..")

        # create a dataset of (i, variable, rows, cols) for atmospheric river and non atmospheric river
        fname = os.path.join(self.repo_path, self.outfile)

        # hack to avoid issue of unclosed hdf5 file handle in ipython
        try: os.remove(fname)
        except: pass
        f = h5py.File(fname, 'w')
        shape =self.shape
        blk = 1000   # block size, hard coded
        
        # make datasets(group) for AR, Non_AR, Landmask and allow them to be extensible
        one = f.create_dataset('AR', (blk,) + shape,dtype=np.float32,maxshape=((None,) + shape))
        zero = f.create_dataset('Non_AR', (blk,) + shape,dtype=np.float32,maxshape=((None,) + shape))
        one_date=f.create_dataset('AR_date',(blk,),dtype=np.dtype('S15'),maxshape=((None,)))
        zero_date=f.create_dataset('NonAR_date',(blk,),dtype=np.dtype('S15'),maxshape=((None,)))

        # prognostic variables are keys of 'p' below
        if intersect: #if need to calculate the intersection between AR and landmasks
           vs1 = ["TMQ","mask"] 
           vs2 =['date'] 
        else:
           vs1=["TMQ"]
           vs2=["date"] 

        #function to calculate the intersection 
        def overlay(v,bb):
            """
            this function is to calculate the intersection of landmask and AR
            IN:  v: the 2D TMQ field
                 bb: land sea mask 2D field
            OUT: 2D field that AR overlap land
            """
            threshold=20.  # the threshold for TMQ
            inter=np.multiply(v,bb)
            inter_idx=inter < threshold
            inter[inter_idx] = 0
            
            return inter
        
        # function to write data into h5 format   
        def read_write_ar(files, h5dset):
            """
            this function define routine of writting out hdf5 file
            IN: files, data files that need to be transformed to hdf5
                h5dset: defined h5 data chuck 
            """
            i = 0
            r = np.empty(shape, dtype=np.float32)
            for f in files:
                logger.debug('Reading %s' %f)
                with open(f, 'rb') as fid: 
                     p=pickle.load(fid)   
                l = len(p[vs1[0]])
                logger.debug("number of events  %d" %l)

                # loop through AR instances
                # if need to calculate the intersection
                if intersect:  
                   for j in range(l):
                        """
                        # old routine to write multiple variables
                        for m,k in zip(vs1, range(len(vs1))):
                            r[k]=p[m][j]
                        """
                        olap=overlay(p[vs1[0]][j],landsea)
                        r[0]=p[vs1[0]][j]
                        r[1]=olap
                        h5dset[i] = r
                        i += 1

                        #check block dimension
                        if i == maxsize: break
                        if i % 100 == 0:
                           logger.debug('%d' % i)
                        # resize hdf5 as necessary to make room for more rows
                        if i % blk == 0:
                           h5dset.resize(i+blk, axis=0)
                #if not calculate the intersection
                else:
                   for j in range(l):
                        for m,k in zip(vs1, range(len(vs1))):
                            r[k]=p[m][j]
                            h5dset[i] =r     
     
                        #check block dimension
                        if i == maxsize: break
                        if i % 100 == 0:
                           logger.debug('%d' % i)
                        # resize hdf5 as necessary to make room for more rows
                        if i % blk == 0:
                           h5dset.resize(i+blk, axis=0)
                #if max row reached
                if i ==maxsize: break
            h5dset.resize(i, axis=0)
            logger.debug('Wrote %d records in total.' % i)
     
        #write date information into h5   
        def read_write_date(files, h5dset):
            """
            this function define routine of writting time information into hdf5 file
            IN: files, data file that need to be transformed to hdf5
                h5dset, defined h5 data block
            """
            i = 0
            r = np.empty((1,), dtype=np.dtype('S15'))
            for f in files:
                logger.debug('Reading %s' %f)
                with open(f, 'rb') as fid:   
                     p=pickle.load(fid)   
                l = len(p[vs2[0]])
                print("number of events  %d" %l)
                # loop through AR instances
                for j in range(l):
                     # loop through prognostic variables fo reach case
                     for m,k in zip(vs2, range(len(vs2))):
                         r[k]=p[m][j]
                     h5dset[i] = r
                     i += 1

                     #check block dimension
                     if i == maxsize: break
                     if i % 100 == 0:
                        logger.debug('%d' % i)
                     # resize hdf5 as necessary to make room for more rows
                     if i % blk == 0:
                        h5dset.resize(i+blk, axis=0)
                if i ==maxsize: break


            # final resize
            h5dset.resize(i, axis=0)
            logger.debug('Wrote %d records in total.' % i)

        # write pkl to HDF5
        read_write_ar(pf, one)
        read_write_ar(nf, zero)
        read_write_date(pf,one_date)
        read_write_date(nf,zero_date)
        f.close()

#not sure what this is for
if __name__ == '__main__':
    import os

    # create HDF5 version of dataset

    p = "/global/project/projectdirs/mantissa/climate/Yunjie/ar_patch2/us_new_samesize/"
    fname="atmosphericriver_eu+TMQ_Sep3.h5"
    h = ar(repo_path=p, case="US",shape=(2,148,224),intersect=True, outfile=fname)
    h.hdf5(maxsize=10000)
