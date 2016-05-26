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
from pyspark import SparkContext

caffe_output_file = sys.argv[1]
plot_dir = sys.argv[2]
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
sc = SparkContext()

lines = sc.textFile(caffe_output_file)
lines.cache()
loss = lines.filter(lambda x: ', loss =' in x )\
    .map(lambda y: y.split(']')[-1].split())\
    .map(lambda z: (int(z[1][:-1]), float(z[-1])))
losses = np.asarray(loss.collect())

accuracies = lines.filter(lambda x: 'seg-accuracy =' in x ).\
    map(lambda y: float(y.split('= ')[-1]))

acc = np.asarray(accuracies.collect())
plt.figure(1)
plt.plot(losses[:,0], losses[:,1])
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig(plot_dir + '/loss_curve.jpg')

plt.clf()
plt.figure(1)
plt.plot(np.arange(acc.shape[0]),acc)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig(plot_dir + '/accuracy_curve.jpg')

