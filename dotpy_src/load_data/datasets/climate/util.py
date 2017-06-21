


import sys
import os
import numpy as np
import threading
import netCDF4 as nc
from netCDF4 import MFDataset
from os import listdir, system
from os.path import isfile, join, isdir
import imp
import itertools
import time
import inspect
import copy



def get_files_from_dir(dir_,type_, years):
    final_dir = join(dir_,type_)
    return [join(final_dir, fil) for fil in listdir(final_dir) if any([year in fil for year in years])]

def get_files_from_list(list_dir, type_, images_or_labels_or_crop_indices):
    with open(join(list_dir, type_ + "_" + images_or_labels_or_crop_indices + "_files.txt"), "r") as f:
        return [line.strip("\n") for line in f.readlines()]

