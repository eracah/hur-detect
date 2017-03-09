


import sys
import os



from dataset import make_datasets
import numpy as np



import threading



from configs import configs



def get_generator(typ):
    num_ims = configs["num_"+ typ+"_ims"]
    return ClimateGenThreadSafe(batch_size = configs["batch_size"],
                                typ=typ, 
                                tf_mode=True, 
                                num_ex = num_ims) 
    




# def climate_gen(batch_size=128, typ="tr", tf_mode=True, num_ex=-1):
#     cl_data = make_datasets(num_tr_examples=num_ex,typ=typ)
#     data = getattr(cl_data, typ)
    
    
#     #supposed to go indefinitely
#     while 1:
#         ims,lbls = data.next_batch(batch_size=batch_size)
#         if tf_mode:
#             ims, lbls  = np.transpose(ims,axes= (0,2,3,1)), np.transpose(lbls,axes= (0,2,3,1))
#         lbls = correct_class_labels(lbls)
        
#         yield ims, lbls
            
                



class ClimateGenThreadSafe(object):
    def __init__(self, batch_size=128, typ="tr", tf_mode=True, num_ex=-1):
        self.data = make_datasets(num_examples=num_ex,typ=typ)
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
            lbls = correct_class_labels(lbls)
            return ims, lbls



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
    



if __name__ == "__main__":
    cg = ClimateGenThreadSafe(batch_size=4)
    for im,lbl in cg:
        print im.shape, lbl.shape
        
    # for im,lbl in climate_gen(batch_size=10, num_tr=20):
    #         print im.shape, lbl.shape
    #         print np.any(lbl[:,:,:,5] == 4)








