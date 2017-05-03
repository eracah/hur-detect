


import sys
import os
from os import listdir, system
from os.path import isfile, join, isdir
import numpy as np
import imp
import itertools
import os
import sys
import time
import inspect
import copy
import random
import collections

import importlib



#Thanks to TensorFlow (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py)
#for inspiration for this data structure
class BatchFetcher(object):

    def __init__(self, images, labels, num_examples=-1):
        """Create a Data Structure that can continually
        return minibatches of some data

        Keyword arguments:
        images -- an object that can be indexed (aka an array or a class with the __get_item__ fxn implemented)
                    * also the function shuffle and the attribute total_examples must be implemented
        labels -- same as above
        """


        self._images = images
        self._labels = labels
        self._num_examples = num_examples if num_examples != -1 else images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def shuffle(self):
        pass

    def next_batch(self, batch_size, shuffle=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            raise NotImplementedError
            
            #shuffle files
            # no shuffle for now
#             seed = np.random.randint(0,1000)
#             self._images.shuffle(seed)
#             self._labels.shuffle(seed)

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                raise NotImplementedError
                
#                 seed = np.random.randint(0,1000)
#                 self._images.shuffle(seed)
#                 self._labels.shuffle(seed)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            

            return np.concatenate((images_rest_part, images_new_part), axis=0),                    np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
           
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            images = self._images[start:end]
            labels = self._labels[start:end]
            #print images.shape
            #print labels.shape
            return images,labels





