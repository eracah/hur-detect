__author__ = 'racah'
import h5py
import numpy as np
from operator import mul
import os
import glob
import sys
import time
from multiprocessing import Pool
from functools import partial








#1 0 is hur
#0 1 is nhur
class LoadHurricane():
    def __init__(self,batch_size=None, flatten=False, num_ims=None, seed=4):
        self.seed = seed
        self.batch_size = batch_size
        self.num_ims = num_ims
        self.seed = seed
        self.flatten=flatten
    #TODO try on 96x96 (use bigger file -> get from cori)
    def load_hurricane(self, path):

        print 'getting data...'
        h5f = h5py.File(path)
        hurs = h5f['hurricane'][:]
        nhurs = h5f['nothurricane'][:]
        hurs_bboxes = np.asarray(h5f['hurricane_box']).reshape(hurs.shape[0],4)
        nhurs_bboxes = np.zeros((nhurs.shape[0],4))
        #input_size = hurs.shape[0] + nhurs.shape[0]
        #shp = ((hurs.shape[0]+nhurs.shape[0],) + hurs.shape[1:])

        inputs = np.vstack((hurs,nhurs))
        # tmpfile= os.path.basename(path + '/tst.h5')
        # h5tmp = h5py.File(tmpfile)
        # # if'inputs' not in h5tmp:
        # inputs =h5tmp.create_dataset('inputs', shape=shp, dtype='f4')
        # inputs[:hurs.shape[0]] = hurs
        # inputs[hurs.shape[0]:] = nhurs
        # # else:
        # #     inputs = h5tmp['inputs']

        bboxes = np.vstack((hurs_bboxes,nhurs_bboxes))
        cl_labels = np.zeros((inputs.shape[0], 2))
        cl_labels[:hurs.shape[0],0] = 1.
        cl_labels[hurs.shape[0]:,1] = 1.

        if not self.num_ims:
            self.num_ims = inputs.shape[0]


        else:
            self.num_ims = self.num_ims

        print self.num_ims


        hur_masks = self.gen_masks(inputs[:self.num_ims],bboxes)
        self.y_dims = hur_masks.shape[1:]
        #hur_masks = hur_masks.reshape(hur_masks.shape[0], np.prod(hur_masks.shape[1:]))
        tr_i, te_i, val_i = self.get_train_val_test_ix(self.num_ims)

        return self.set_up_train_test_val(inputs, bboxes, hur_masks,cl_labels, tr_i, te_i, val_i)



    def get_train_val_test_ix(self, num_ims):
        # tr, te, val is 0.6,0.2,0.2
        ix = range(num_ims)

        #make sure they are all multiple of batch size
        n_te = int(0.2*num_ims)
        n_val = int(0.25*(num_ims - n_te))
        n_tr =  num_ims - n_te - n_val

        if self.batch_size:
            n_te -= n_te % self.batch_size
            n_val = self.batch_size * ((n_val) / self.batch_size)
            n_tr =  self.batch_size * ((n_tr) / self.batch_size)


        #shuffle once deterministically
        np.random.RandomState(3).shuffle(ix)
        te_i = ix[:n_te]
        rest = ix[n_te:]

        np.random.RandomState(self.seed).shuffle(rest)
        val_i = rest[:n_val]
        tr_i = rest[n_val:n_val + n_tr]
        return tr_i, te_i, val_i


    def set_up_train_test_val(self,hurs, boxes,hur_masks, cl_labels, tr_i,te_i, val_i):

        x_tr, y_tr, bbox_tr, lbl_tr = hurs[tr_i], hur_masks[tr_i], boxes[tr_i], cl_labels[tr_i]
        x_tr, tr_means, tr_stds = self.normalize_each_channel(x_tr)
        self.test_masks(bbox_tr, y_tr,np.random.randint(x_tr.shape[0]))
        x_te, y_te, bbox_te, lbl_te = hurs[te_i], hur_masks[te_i], boxes[te_i], cl_labels[te_i]
        x_te, _ ,_ = self.normalize_each_channel(x_te, tr_means, tr_stds)

        self.test_masks(bbox_te, y_te,np.random.randint(x_te.shape[0]))
        x_val, y_val, bbox_val, lbl_val = hurs[val_i], hur_masks[val_i], boxes[val_i], cl_labels[val_i]
        x_val, _ ,_ = self.normalize_each_channel(x_val, tr_means, tr_stds)
        self.test_masks(bbox_val, y_val,np.random.randint(x_val.shape[0]))

        if self.flatten:
            x_tr = x_tr.reshape(x_tr.shape[0], reduce(mul, x_tr.shape[1:]))
            x_te = x_te.reshape(x_te.shape[0], reduce(mul, x_te.shape[1:]))
            x_val = x_val.reshape(x_val.shape[0], reduce(mul, x_val.shape[1:]))

        x_dims = hurs.shape[1:]

        return {'tr': (x_tr, y_tr, bbox_tr, lbl_tr), \
        'te':(x_te, y_te, bbox_te, lbl_te), \
        'val': (x_val, y_val, bbox_val, lbl_val)}
        # return {'x_train': x_tr, 'y_train': y_tr, 'x_test': x_te, 'y_test': y_te,'x_val':x_val, 'y_val':y_val, 'boxes': boxes}

    def normalize_each_channel(self,arr, means=[], stds=[]):
        # assumes channels are on the axis 1
        if len(means) == 0:
            means = np.mean(arr, axis=(0, 2, 3))
        if len(stds) == 0:
            stds = np.std(arr, axis=(0, 2, 3))
        for channel, (mean, std) in enumerate(zip(means, stds)):
            arr[:, channel, :, :] -= mean
            arr[:, channel, :, :] /= std
        return arr, means, stds


    def test_masks(self, box,mask, ind=0):
        b = box[ind]
        m = mask[ind]
        section = m[:,b[0]:b[2], b[1]:b[3]]
        assert np.all(section[0,:] == 1.), section[0]
        assert np.all(section[1,:] == 0.)


    def gen_masks(self,hurs,bboxes):
        '''

        :param hurs: n_hurs x 8 x H x W array
        :param bboxes: n_hurs x 4 array
        :return:
            n_hurs x 2 x H x W array
                where the first cahnnel is p(hurricane) and second channel is p(not hurricane)

        '''
        t = time.time()
        p_hur = np.zeros((hurs.shape[0], hurs.shape[2],hurs.shape[3]))
        p_nhur = np.ones((hurs.shape[0], hurs.shape[2], hurs.shape[3]))

        #TODO: vectorize
        for i in range(hurs.shape[0]):
            bbox=bboxes[i]
            p_hur[i, bbox[0]:bbox[2],bbox[1]:bbox[3]] = 1.
            p_nhur[i, bbox[0]:bbox[2],bbox[1]:bbox[3]] = 0.
        print "gen_masks took: %5.2f seconds"%(time.time()-t)
        hur_masks = np.hstack((p_hur, p_nhur)).reshape(hurs.shape[0], 2, hurs.shape[2], hurs.shape[3])
        return hur_masks


if __name__ == "__main__":
    pass
    # path = sys.argv[1]
    # preproc_data_dir='./preproc_data'
    # load_hurricane(path, 1, 1, True, preproc_data_dir)








