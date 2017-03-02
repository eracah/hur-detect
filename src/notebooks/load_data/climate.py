
# coding: utf-8

# In[5]:


import sys
import os
from nbfinder import NotebookFinder
sys.meta_path.append(NotebookFinder())
from dataset import make_datasets
import numpy as np


# In[6]:

import threading


# In[ ]:

def get_generator():
    return ClimateGenThreadSafe
    


# In[7]:


def climate_gen(batch_size=128, typ="tr", tf_mode=True, num_tr=-1):
    cl_data = make_datasets(num_tr_examples=num_tr)
    data = getattr(cl_data, typ)
    
    
    #supposed to go indefinitely
    while 1:
        ims,lbls = data.next_batch(batch_size=batch_size)
        if tf_mode:
            ims, lbls  = np.transpose(ims,axes= (0,2,3,1)), np.transpose(lbls,axes= (0,2,3,1))
        lbls = correct_class_labels(lbls)
        
        yield ims, lbls
            
                


# In[18]:

class ClimateGenThreadSafe(object):
    def __init__(self, batch_size=128, typ="tr", tf_mode=True, num_tr=-1):
        cl_data = make_datasets(num_tr_examples=num_tr)
        self.data = getattr(cl_data, typ)
        self.i = 0
        self.tf_mode = tf_mode
        self.batch_size = batch_size
        # create a lock
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        # acquire/release the lock when updating self.i
        with self.lock:
            ims,lbls = self.data.next_batch(batch_size=self.batch_size)
            if self.tf_mode:
                ims, lbls  = np.transpose(ims,axes= (0,2,3,1)), np.transpose(lbls,axes= (0,2,3,1))
            lbls = correct_class_labels(lbls)
            return ims, lbls


# In[19]:

def correct_class_labels(lbls, tf_mode=True):
    """subtract class labels by 1, so labels used to be 1-4 now 0-3 and 0 is still 0"""
    if tf_mode:
        cl_index = 5
        lbls[:,:,:,cl_index] = lbls[:,:,:,cl_index] - 1
        lbls[:,:,:,cl_index] = np.where(lbls[:,:,:,cl_index]==-1,
                                        np.zeros_like(lbls[:,:,:,cl_index]),
                                        lbls[:,:,:,cl_index] )
    else:
        assert False, "not implemented"
    return lbls
    


# In[20]:

if __name__ == "__main__":
    cg = ClimateGenThreadSafe(batch_size=4)
    for im,lbl in cg:
        print im.shape, lbl.shape
        
    # for im,lbl in climate_gen(batch_size=10, num_tr=20):
    #         print im.shape, lbl.shape
    #         print np.any(lbl[:,:,:,5] == 4)


# In[ ]:




# In[ ]:



