
import matplotlib; matplotlib.use("agg")


import h5py
import numpy as np
import os



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



path = '/project/projectdirs/dasrepo/gordon_bell/climate/data/detection/'



classif_file = h5py.File(path + 'hur_class.h5')



detect_file = h5py.File(path + 'hur_detect.h5')



train_file = h5py.File(path + 'hur_train_val.h5')

detect_file['hurricanes'].shape

detect_file.keys()

tr_hurs = train_file.create_dataset('hurs', shape=(20000,8,96,96))

tr_hurs[:15000] = detect_file['hurricanes']

tr_hurs[15000:] = classif_file['hurricanes'][:5000]

tr_hur_boxes = train_file.create_dataset('hur_boxes', shape=(20000,1,4))

tr_hur_boxes[:15000] = detect_file['hurricane_boxes']

tr_hur_boxes[15000:] = classif_file['hurricane_boxes'][:5000]

tr_nhurs = train_file.create_dataset('nhurs', shape=(20000,8,96,96))

tr_nhurs[:15000] = detect_file['not_hurricanes']

tr_nhurs[15000:] = classif_file['not_hurricanes'][:5000]

train_file.close()



os.remove(path + 'hur_test.h5')



test_file = h5py.File(path + 'hur_test.h5')



te_hurs = test_file.create_dataset('hurs', shape=(5000,8,96,96))



te_hurs[:5000] = classif_file['hurricanes'][5000:]

te_hur_boxes = test_file.create_dataset('hur_boxes', shape=(5000,1,4))

te_hur_boxes[:5000] = classif_file['hurricane_boxes'][5000:]

te_nhurs = test_file.create_dataset('nhurs', shape=(5000,8,96,96))



te_nhurs[:5000] = classif_file['not_hurricanes'][5000:]







os.listdir(path)



test_file.close()



test_file['hur_boxes'].shape



path






train_file = h5py.File(path + 'hur_train_val.h5')



test_file['hurs'][100,1]



classif_file['hurricanes'][9999,0]





