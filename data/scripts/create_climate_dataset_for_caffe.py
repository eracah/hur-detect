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
if len(sys.argv) < 4:
    print "usage: %s %s %s %s" % (sys.argv[0], 'h5_path', 'desired_h5save_dir', 'desired_txt_save_dir' 'use_negative_images (optional)')
    sys.exit()
h5_path =  os.path.abspath(os.path.expandvars(sys.argv[1]))
des_h5dir = os.path.abspath(os.path.expandvars(sys.argv[2]))
des_txtdir = os.path.abspath(os.path.expandvars(sys.argv[3]))
use_neg = None
if len(sys.argv) > 4:
    use_neg = sys.argv[4]

l = LoadHurricane()
data_dict = l.load_hurricane(h5_path, use_neg)

for dtype in ['tr','te','val']:
    x, mask, bbox, cl_lbl = data_dict[dtype]

    filename = des_h5dir + '/' + 'hur' + ('_w_neg' if use_neg else '') + '/hur_' + dtype  + '.h5'
    dirname = os.path.dirname(filename)
    print dirname
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    a = h5py.File(filename)
    a.create_dataset('data', data=x)
    a.create_dataset('mask', data=mask)
    a.create_dataset('bbox', data=bbox)
    a.create_dataset('label', data=cl_lbl)
    a.close()

    txtdir = des_txtdir + '/' + 'hur' + ('_w_neg' if use_neg else '')
    if not os.path.exists(txtdir):
        os.mkdir(txtdir)

    with open(txtdir + '/' + 'hur_' + dtype + '.txt', 'a+') as f:
        f.write(filename)

