
import matplotlib; matplotlib.use("agg")


import h5py



h5path = '/project/projectdirs/nervana/evan/preproc_data/data_for_caffe/hurricanes.h5'



train = '/project/projectdirs/nervana/evan/preproc_data/data_for_caffe/hurricane_train.h5'



new_h5 = '/project/projectdirs/nervana/evan/preproc_data/data_for_caffe/hurricanes_test.h5'



thf = h5py.File(train)



nhf = h5py.File(new_h5)



hf = h5py.File(h5path)



examples = hf['label'].shape[0]



ntr = int(0.8 * examples)



nhf.create_dataset('data', data=hf['data'][ntr:])



nhf.create_dataset('label', data=hf['label'][ntr:])



hfd = hf['data'][:ntr]



hfl = hf['label'][:ntr]



thf.create_dataset('data', data=hfd)



thf.create_dataset('label', data=hfl)



thf.close(); nhf.close(); hf.close()






















ls /project/projectdirs/nervana/evan/preproc_data/data_for_caffe/





