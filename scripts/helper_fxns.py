
import matplotlib; matplotlib.use("agg")


import numpy as np
import os
import sys
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

                
class early_stop(object):
    def __init__(self, patience=500):
        self.patience = patience   # look as this many epochs regardless
        self.patience_increase = 2  # wait this much longer when a new best is
                                      # found
        self.improvement_threshold = 0.995  # a relative improvement of this much is
                                      # considered significant
        self.validation_frequency = self.patience // 2
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        self.best_validation_loss = np.inf

    def keep_training(self, val_loss, epoch):
        print epoch
        print val_loss
        print self.best_validation_loss
        if val_loss < self.best_validation_loss:
                #improve patience if loss improvement is good enough
                if val_loss < self.best_validation_loss *                     self.improvement_threshold:
                    self.patience = max(self.patience, epoch * self.patience_increase)

                self.best_validation_loss = val_loss
        if self.patience <= epoch:
            return False
        else:
            return True


    







def create_run_dir():
    results_dir = './results'
    run_num_file = os.path.join(results_dir, "run_num.txt")
    if not os.path.exists(results_dir):
        print "making results dir"
        os.mkdir(results_dir)

    if not os.path.exists(run_num_file):
        print "making run num file...."
        f = open(run_num_file,'w')
        f.write('0')
        f.close()




    f = open(run_num_file,'r+')

    run_num = int(f.readline()) + 1

    f.seek(0)

    f.write(str(run_num))


    run_dir = os.path.join(results_dir,'run%i'%(run_num))
    os.mkdir(run_dir)
    return run_dir





