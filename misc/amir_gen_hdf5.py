"""
Provides neon.datasets.Dataset class for hurricane patches data

To generate hdf5 file:

python hurricanes.py
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas
import h5py
import ipdb
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Hurricane(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if 'repo_path' not in kwargs:
            raise ValueError('Missing repo_path.')

        # location of pkl hurricane files and hdf5
        self.pkldir = '/global/scratch2/sd/syoh/hurricane'  # Sang's directory

        # Amir's scratch directory
        #self.rootdir = os.path.join(self.repo_path, self.__class__.__name__)
        self.scratchdir = '/scratch2/scratchdirs/khosra/neon-data'
        self.rootdir = os.path.join(self.scratchdir, self.__class__.__name__)        

    def hdf5(self, maxsize=1000, outfile='hurricanes.h5'):
        """
        Convert pickle files into a single HDF5 containing two datasets:

         \1: hurricanes        (-1, 8, 33, 33) with 8 prognostic variables
         \0: random snapshots  (-1, 8, 33, 33) 
        """
        # get a list of files in the hurricane directory
        pf = glob.glob(os.path.join(self.pkldir, 'hurricanes*_imgs.pkl'))
        nf = glob.glob(os.path.join(self.pkldir, 'not_hurricanes*_imgs.pkl'))

        # create a dataset of (i, variable, rows, cols) for hurricanes and not hurricanes
        fname = os.path.join(self.rootdir, outfile)

        # hack to avoid issue of unclosed hdf5 file handle in ipython
        try: os.remove(fname)
        except: pass
        
        f = h5py.File(fname, 'w')

        shape = (8, 32, 32)
        blk = 10000   # block size

        # make datasets and allow them to be extensible
        one = f.create_dataset('1', (blk,) + shape,
                               dtype=np.float32,
                               maxshape=((None,) + shape))
        zero = f.create_dataset('0', (blk,) + shape,
                                dtype=np.float32,
                                maxshape=((None,) + shape))

        # prognostic variables are keys of 'p' below
        vs = (u'TMQ', u'V850', u'PSL', u'U850',
              u'T500', u'UBOT', u'T200', u'VBOT')


        def read_write(files, h5dset):
            """
            Read list of pandas files and write to single h5 dataset.
            """

            i = 0
            r = np.empty(shape, dtype=np.float32)
            for f in files:
                logger.debug('Reading %s' % f)
                p = pandas.read_pickle(f)
                l = len(p[vs[0]])
                
                # loop through hurricane instances
                for j in range(l):

                    # loop through prognostic variables
                    for v,k in zip(vs, range(len(vs))):
                        r[k] = p[v][j][:32,:32]

                    h5dset[i] = r
                    i += 1

                    if i == maxsize: break
                    if i % 100 == 0:
                        logger.debug('%d' % i)
                        
                    # resize hdf5 as necessary to make room for more rows
                    if i % blk == 0:
                        h5dset.resize(i+blk, axis=0)
                        
                if i == maxsize: break

            # final resize
            h5dset.resize(i, axis=0)
            logger.debug('Wrote %d records in total.' % i)

        read_write(pf, one)
        read_write(nf, zero)

        f.close()

if __name__ == '__main__':
    import os

    # create HDF5 version of dataset
    p = os.path.join(os.path.expanduser('~'), 'nervana', 'neon-data')
    h = Hurricane(repo_path=p)
    h.hdf5(maxsize=1000000, outfile='hurricanes_all.h5')
    
