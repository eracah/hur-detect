


import sys
import os
import numpy as np
import threading

if __name__ == "__main__":
    sys.path.append("../../../")



class GenThreadSafe(object):
    def __init__(self, dataset, shape=None, batch_size=128, typ="tr", tf_mode=False, num_ex=-1, make_label_fxn=None):
        self.data = dataset
        self.tf_mode = tf_mode
        self.batch_size = batch_size
        # create a lock
        self.lock = threading.Lock()

        self.make_label_fxn = make_label_fxn

            
        
    def get_make_label_fxn(self, all_boxes):
        pass
        
    def __iter__(self):
        return self
   
    @property
    def num_ims(self):
        return self.data.num_examples
    
    def next(self):
        
        # acquire/release the lock when updating self.i
        with self.lock:
            ims,lbls = self.data.next_batch(batch_size=self.batch_size)
            if self.tf_mode:
                ims = np.transpose(ims,axes=(0,2,3,1))
            return ims, lbls

