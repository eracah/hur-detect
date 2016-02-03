#!/usr/bin/env python
__author__ = 'racah'

import h5py
import numpy as np
from operator import mul
import os
import glob
import sys
import time
from data_loader import LoadHurricane


def create_dataset(h5f,x, mask, bbox, cl_lbl, imin=None, imax=None):
    if imin==None and imax==None:
        imin=0
        imax=x.shape[0]

    h5f.create_dataset('data', data=x[imin:imax])
    h5f.create_dataset('mask', data=mask[imin:imax])
    h5f.create_dataset('bbox', data=bbox[imin:imax])
    h5f.create_dataset('label', data=cl_lbl[imin:imax])
    h5f.close()




if len(sys.argv) < 4:
    print "usage: %s %s %s %s %s %s" % (sys.argv[0], 'h5_path', 'desired_h5save_dir', 'desired_txt_save_dir', 'max size of eac h5 file' ,'use_negative_images (optional)')
    sys.exit()
h5_path = os.path.abspath(os.path.expandvars(sys.argv[1]))
des_h5dir = os.path.abspath(os.path.expandvars(sys.argv[2]))
des_txtdir = os.path.abspath(os.path.expandvars(sys.argv[3]))
max_size_of_h5_files = None

if len(sys.argv) > 4:
    max_size_of_h5_files = sys.argv[4]
    if max_size_of_h5_files == 'None':
        max_size_of_h5_files = None
    else:
        max_size_of_h5_files = int(max_size_of_h5_files)
use_neg = None
if len(sys.argv) > 5:
    use_neg = sys.argv[5]

l = LoadHurricane()
data_dict = l.load_hurricane(h5_path, use_neg)

for dtype in ['tr','te','val']:
    x, mask, bbox, cl_lbl = data_dict[dtype]

    base_hur_dirname = 'hur' + ('_split' if max_size_of_h5_files else '') +  ('_w_neg' if use_neg else '')
    filename = des_h5dir + '/' + base_hur_dirname + '/hur_' + dtype
    dirname = os.path.dirname(filename)
    print dirname
    if not os.path.exists(dirname):
        os.mkdir(dirname)


    txtdir = des_txtdir + '/' + base_hur_dirname
    if not os.path.exists(txtdir):
            os.mkdir(txtdir)

    if max_size_of_h5_files >= x.shape[0] or max_size_of_h5_files is None:
        h5f = h5py.File(filename + '.h5')
        create_dataset(h5f, x, mask, bbox, cl_lbl, imin=0, imax=x.shape[0])
        with open(txtdir + '/' + 'hur_' + dtype + '.txt', 'a+') as f:
            f.write(filename + '.h5' + '\n')

    else:
        for n, i in enumerate(range(0, x.shape[0], max_size_of_h5_files)):
            offset = max_size_of_h5_files
            fname = filename + '_' + str(n) + '.h5'
            h5f = h5py.File(fname)
            if i + offset <= x.shape[0]:
                imax = i + offset
            else:
                imax = x.shape[0]
            create_dataset(h5f, x, mask, bbox, cl_lbl, imin=i, imax=imax)

            with open(txtdir + '/' + 'hur_' + dtype + '.txt', 'a+') as f:
                f.write(fname + '\n')

with open(dirname + '/README.txt', 'w') as b:
    b.write('Each h5 files contain four datasets:\n'
            'data: n_imagesx8xHxW mean centered patches containing' +( 'or not containing' if use_neg else '') + 'hurricanes\n'
             'label: n_images x 1 array 1 means hurricane 0 means not\n'
             'mask: n_imagesx1xHxW array where a value of 1 mans hurricane at that location and 0 means not\n'
             'bbox: n_imagesx4 array where elements 1 and 3 are the hmin,hmax and 0 and 2 are wmin,wmax of the patch contiang hurricane\n')




