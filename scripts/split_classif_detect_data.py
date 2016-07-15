
import matplotlib; matplotlib.use("agg")


import h5py
import numpy as np

hur_file = h5py.File('/global/project/projectdirs/nervana/yunjie/dataset/localization/larger_hurricanes_loc.h5')

classif_file =h5py.File('/global/project/projectdirs/nervana/evan/detection_data/hur_class.h5')

detect_file = h5py.File('/global/project/projectdirs/nervana/evan/detection_data/hur_detect.h5')

rng = np.random.RandomState(6)

print hur_file.keys()

hurs = hur_file['hurricane'][:]

h_box = hur_file['hurricane_box'][:]

nhurs = hur_file['nothurricane'][:]

inds = np.arange(hurs.shape[0])

rng.shuffle(inds)

hurs = hurs[inds]

h_box = h_box[inds] #forgot this one (very important)

classif_hurs = hurs[:int(0.4*hurs.shape[0])]

detect_hurs = hurs[int(0.4*hurs.shape[0]):]

classif_hbox = h_box[:int(0.4*hurs.shape[0])]
detect_hbox = h_box[int(0.4*hurs.shape[0]):]

detect_hurs.shape

nhurs = nhurs[inds]

classif_nhurs = nhurs[:int(0.4*nhurs.shape[0])]

detect_nhurs = nhurs[int(0.4*nhurs.shape[0]):]

classif_file.create_dataset('hurricanes', data=classif_hurs)

classif_file.create_dataset('hurricane_boxes', data=classif_hbox)

classif_file.create_dataset('not_hurricanes', data=classif_nhurs)

detect_file.create_dataset('hurricanes', data=detect_hurs)

detect_file.create_dataset('hurricane_boxes', data=detect_hbox)

detect_file.create_dataset('not_hurricanes', data=detect_nhurs)

detect_file.close()
classif_file.close()





