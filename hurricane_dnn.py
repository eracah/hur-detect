__author__ = 'racah'

import numpy as np
from data_loader import load_hurricane, is_hurricane
from neon.data import DataIterator, load_cifar10
from custom_initializers import HeWeightInit, AveragingFilterInit
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification, Tanh
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.initializers import Constant, Gaussian, GlorotUniform
from neon.models import Model
from neon.callbacks.callbacks import Callbacks, LossCallback, MetricCallback
from neon.util.argparser import NeonArgparser
from data_loader import get_val_im_size
import os
import sys
import h5py
import pickle
import matplotlib

# 01 is hur 10 is nhur
matplotlib.use('agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches

dirs = {'model_files_dir': './model_files', 'final_dir': './results', 'images_dir': './images'}
for dir in dirs.values():
    if not os.path.exists(dir):
        os.mkdir(dir)

parser = NeonArgparser(__doc__)

# #so we can set batch size as multiple of number of examples per val image we parse arguments
# preproc_data_dir = sys.argv[sys.argv.index('--preproc_data_dir') + 1]
# h5path = sys.argv[sys.argv.index('--h5file') + 1]
# pixels_per_val_image = get_val_im_size(h5path, preproc_data_dir)
# pixels_per_val_image / batch_size

new_args = ['--load_data_from_disk','--num_train', '--num_test_val', '--preproc_data_dir', '--h5file', '--no_model_file']
for new_arg in new_args:
    parser.add_argument(new_arg)

parser.set_defaults(batch_size=1000, test=False, serialize=2, epochs=100, progress_bar=True, datatype='f64',
                    model_file='', load_data_from_disk=1, num_train=6, num_test_val=2, evaluation_freq=1,no_model_file=False,
                    h5file="/global/project/projectdirs/nervana/yunjie/dataset/localization_test/expand_hurricanes_loc.h5",
                    preproc_data_dir='/global/project/projectdirs/nervana/evan/preproc_data')

args = parser.parse_args()
args.load_data_from_disk = bool(int(args.load_data_from_disk))

data_dict = \
    load_hurricane(path=args.h5file,
                   num_train=int(args.num_train),
                   num_test_val=int(args.num_test_val),
                   load_from_disk=args.load_data_from_disk,
                   preproc_data_dir=args.preproc_data_dir)

print "got data!"
w_size = data_dict['w_size']
nclass = data_dict['nclass']
rad = w_size / 2
y_val = data_dict['y_val']
train_set = DataIterator(data_dict['x_train'], data_dict['y_train'], nclass=nclass, lshape=( 8, w_size, w_size))
valid_set = DataIterator(data_dict['x_val'], data_dict['y_val'], nclass=nclass, lshape=(8, w_size, w_size))



w_init = HeWeightInit()
opt_gdm = GradientDescentMomentum(learning_rate=0.01, momentum_coef=0.9)
conv = dict(strides=1, init=w_init, bias=Constant(0), activation=Rectlin())  # , batch_norm=True)

# 13 layer architecure from Ciseran et al. Mitosis Paper adjusted to fit our window size
# (they go from 101x101 as inout to 2x2 as output from last pooling layer), so we adjust to fit the output requirements
layers = [Conv((2, 2, 16), **conv),
          Pooling((2, 2), strides=2),
          Conv((3, 3, 16), **conv),
          Pooling((2, 2), strides=2),
          Conv((3, 3, 16), **conv),
          Pooling((2, 2), strides=2),
          Affine(nout=100, init=w_init, activation=Rectlin(), bias=Constant(1)),
          Affine(nout=2, init=w_init, bias=Constant(0), activation=Softmax())
]

cost = GeneralizedCost(costfunc=CrossEntropyMulti())
mlp = Model(layers=layers)
model_key = '{0}_{1}_{2}'.format(''.join(
    [((l.name[0] if l.name[0] != 'L' else 'FC') if 'Bias' not in l.name and 'Activation' not in l.name else '') +
     ('_'.join([str(num)
                for num in l.fshape]) if 'Pooling' in l.name or 'Conv' in l.name or 'Deconv' in l.name else '')
     for l in mlp.layers.layers]),
                                 train_set.ndata,
                                 '_'.join(map(str, train_set.shape)))

callbacks = Callbacks(mlp, train_set, args, eval_set=valid_set)
h5fin = h5py.File(os.path.join(dirs['final_dir'], model_key + '-final.h5'), 'w')

# add callback for calculating train loss every every <eval_freq> epoch
callbacks.add_callback(
    LossCallback(h5fin.create_group('train_loss'), mlp, eval_set=train_set, epoch_freq=args.evaluation_freq))
callbacks.add_callback(
    LossCallback(h5fin.create_group('valid_loss'), mlp, eval_set=valid_set, epoch_freq=args.evaluation_freq))
callbacks.add_callback(MetricCallback(mlp, eval_set=train_set, metric=Misclassification(), epoch_freq=args.evaluation_freq))
callbacks.add_callback(MetricCallback(mlp, eval_set=valid_set, metric=Misclassification(), epoch_freq=args.evaluation_freq))

model_file_path = os.path.join(dirs['model_files_dir'], '%s.pkl' % model_key)

if os.path.exists(args.model_file):
    print "loading model from file!"
    mlp.load_weights(args.model_file)

elif os.path.exists(model_file_path) and not args.no_model_file=='True':
    print "loading model from file!"
    mlp.load_weights(model_file_path)

mlp.fit(train_set, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
pickle.dump(mlp.serialize(), open(model_file_path, 'w'))

print('Misclassification error = %.1f%%' % (mlp.eval(valid_set, metric=Misclassification()) * 100))

probs = mlp.get_outputs(valid_set)
# probs will have shape (X_val.shape[0],2) number of example_images by the output vector of 2
pos_probs = probs[:, 1]  # hust get second column which corresponds to prob of hurricane
# probs.reshape to n_val_im by one input image channel shape so we can have prob map
cropped_ims = data_dict['cropped_ims']
val_i = data_dict['val_i']
boxes = data_dict['boxes']
prob_map = pos_probs.reshape(len(val_i), cropped_ims[0].shape[1], cropped_ims[0].shape[2])
h5fin.create_dataset('Prob_Maps', data=prob_map)
h5fin.close()




# plot
for i in range(len(val_i)):
    v_i = val_i[i]
    plt.figure(1)
    plt.clf()
    pred = plt.subplot(3,1,1)
    pred.imshow(prob_map[i])
    hur_ch = plt.subplot(3,1,2)
    hur_ch.imshow(cropped_ims[v_i, 2, :, :])
    hur_ch.add_patch(patches.Rectangle(
        (boxes[v_i][0, 0] - rad, boxes[v_i][0, 1] - rad),
        boxes[v_i][0, 2] - boxes[v_i][0, 0],
        boxes[v_i][0, 3] - boxes[v_i][0, 1],
        fill=False))
    # pred.add_patch(patches.Rectangle(
    #     (boxes[v_i][0, 0] - rad, boxes[v_i][0, 1] - rad),
    #     boxes[v_i][0, 2] - boxes[v_i][0, 0],
    #     boxes[v_i][0, 3] - boxes[v_i][0, 1],
    #     fill=False))
    gr_truth = plt.subplot(3,1,3)
    x=cropped_ims[0].shape[1]
    y=cropped_ims[0].shape[2]
    gr_truth.imshow(y_val[i*x*y : (i+1)*x*y].reshape(x,y))

    plt.savefig(os.path.join(dirs['images_dir'], '%s-%i.pdf' % (model_key, i)))


#TODO: Add in learning curve
#TODO:

