__author__ = 'racah'
import h5py
import numpy as np
from operator import mul
import os
import glob
import sys
import time
from data_loader import LoadHurricane
h5_path = sys.argv[1]
des_dir = sys.argv[2]

l = LoadHurricane()
data_dict = l.load_hurricane()

for dtype in range ['tr','te','val']:
    x, y, bbox = data_dict[dtype]
    filename = des_dir + '/' + 'hur_' + dtype + '.h5'
    a = h5py.File(filename)
    a.create_dataset('data', data=x)
    a.create_dataset('label', data=y)
    a.create_dataset('bbox', data=bbox)
    a.close()
    with open(des_dir + '/' + 'hur_' + dtype + '.txt', 'a+') as f:
        f.write(filename)

