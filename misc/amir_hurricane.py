"""
Provides neon.datasets.Dataset class for hurricane patches data
"""
import logging
import numpy as np
import h5py
import os
import ipdb

from neon.datasets.dataset import Dataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Hurricane(Dataset):
    """
    Sets up the NERSC Mantissa hurricane dataset.

    Attributes:
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str, optional): where to locally host this dataset on disk
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if 'repo_path' not in kwargs:
            raise ValueError('Missing repo_path.')

        self.rootdir = os.path.join(self.repo_path, self.__class__.__name__)

    def load(self, sl=None):
        """
        Read data from h5 file, assume it's already been created.

        Create training and test datasets from 1 or more prognostic variables.
        """
        f = h5py.File(os.path.join(self.rootdir, self.hdf5_file), 'r')

        one = f['1']
        zero = f['0']

        # [DEBUG] some debug settings
        v = self.variable            # which variable to pick
        tr = self.training_size    # how many training rows * 2
        te = self.test_size        # how many test rows * 2

        # take equal number of hurricanes and non-hurricanes
        if sl is None:
            sl = slice(None, None, 1)
        self.inputs['train'] = np.vstack((one[:tr,v, sl, sl],
                                          zero[:tr,v, sl, sl]))
        
        # one hot encoding required for MLP 
        self.targets['train'] = np.vstack(([[1,0]] * tr,
                                           [[0,1]] * tr))

        # same with test set
        self.inputs['test'] = np.vstack((one[tr:tr+te,v, sl, sl],
                                         zero[tr:tr+te,v, sl, sl]))
        self.targets['test'] = np.vstack(([[1,0]] * te,
                                          [[0,1]] * te))

        f.close()
        
        # flatten into 2d array with rows as samples
        # and columns as features
        dims = np.prod(self.inputs['train'].shape[-2:])
        self.inputs['train'].shape = (-1, dims)
        self.inputs['test'].shape = (-1, dims)

        # shuffle training set
        s = range(len(self.inputs['train']))
        np.random.shuffle(s)
        self.inputs['train'] = self.inputs['train'][s]
        self.targets['train'] = self.targets['train'][s]

        # [DEBUG] shuffle test to create errors
        # s = range(len(self.targets['test']))
        # np.random.shuffle(s)
        # self.targets['test'] = self.targets['test'][s]

        def normalize(x):
            """Make each column mean zero, variance 1"""
            x -= np.mean(x, axis=0)
            x /= np.std(x, axis=0)

        map(normalize, [self.inputs['train'], self.inputs['test']])

        # convert numpy arrays into CPUTensor backend
        self.format()
    
