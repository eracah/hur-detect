#!/usr/bin/env python
__author__ = 'racah'
import h5py
import numpy
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
import os
import numpy as np


class PlotProbMaps(object):
    def __init__(self, hdf5_filepath, plots_dir):

        self.plots_dir = plots_dir
        if not os.path.exists(plots_dir):
            os.mkdir(plots_dir)

        self.h5f = h5py.File(hdf5_filepath)
        h5keys = self.h5f.keys()

        keywords = ['mask', 'data_data', 'seg']#, 'bbox']
        #get first dataset name in hdf5 file that contains one of the keywords above
        self.key_dict = { k : filter(lambda x: k in x, h5keys)[0] for k in keywords }
        print self.key_dict



    def save_side_by_side(self, data, seg, mask, bbox, i, plot_dir):
        plt.figure(1)
        plt.clf()
        pred = plt.subplot(3,1,1)
        pred.imshow(seg[i,0])
        hur_ch = plt.subplot(3,1,2)
        hur_ch.imshow(data[i,3, :, :])
        #self.add_bbox(hur_ch, bbox)
        gr_truth = plt.subplot(3,1,3)
        gr_truth.imshow(mask[i,0])

        plt.savefig(os.path.join(self.plots_dir, '%i.jpg' % ( i)))

    def add_bbox(self, subplot, bbox):
            subplot.add_patch(patches.Rectangle(
            xy=(bbox[0], bbox[1]),
            width=bbox[2] - bbox[0],
            height=bbox[3] - bbox[1],
            fill=False))

    def plot(self, num_ims):
        mask = self.h5f[self.key_dict['mask']]
        data = self.h5f[self.key_dict['data_data']]
        seg = self.h5f[self.key_dict['seg']]
        #bbox = self.h5f[self.key_dict['bbox']]
        bbox=[]


        for i in np.random.random_integers(0,seg.shape[0],num_ims):
            self.save_side_by_side(data, seg, mask, bbox, i, self.plots_dir)


if __name__ == "__main__":
    hdf5_filepath = sys.argv[1]
    plots_dir = sys.argv[2]
    num_ims = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    p = PlotProbMaps(hdf5_filepath, plots_dir)
    p.plot(num_ims)