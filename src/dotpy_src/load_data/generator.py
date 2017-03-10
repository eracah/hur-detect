


import sys
import os

from dataset import make_dataset
import numpy as np

import threading

from configs import configs




class GenThreadSafe(object):
    def __init__(self, dataset, batch_size=128, typ="tr", tf_mode=True, num_ex=-1):
        self.data = dataset
        self.tf_mode = tf_mode
        self.batch_size = batch_size
        # create a lock
        self.lock = threading.Lock()

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
                ims, lbls  = np.transpose(ims,axes= (0,2,3,1)), np.transpose(lbls,axes= (0,2,3,1))
            return ims, lbls

#thread safe
class SemisupWrapper(object):
    def __init__(self,generator):
        self.generator = generator
        self.lock = threading.Lock()
        
    def __iter__(self):
        return self
    
    @property
    def num_ims(self):
        return self.generator.num_ims
    
    def next(self):
        with self.lock:
            ims, lbls = self.generator.next()
            return ims, {"box_score":lbls,"reconstruction":ims}
        

def get_generator(typ, mode="supervised", data_name=None, batch_size=None):
    
    num_ims = configs["num_"+ typ+"_ims"]
    dataset = make_dataset(num_examples=num_ims,typ=typ, data_name=data_name)
    
    if batch_size is None:
        batch_size = configs["batch_size"]
    generator = GenThreadSafe(dataset, 
                              batch_size = batch_size,
                              typ=typ, 
                              tf_mode=True, 
                              num_ex = num_ims) 
    if mode == "semi_supervised":
        generator = SemisupWrapper(generator)
        
    return generator
    


if __name__ == "__main__":
    cg = GenThreadSafe(batch_size=4)
    for im,lbl in cg:
        print im.shape, lbl.shape
        





