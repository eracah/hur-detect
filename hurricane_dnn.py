__author__ = 'racah'
from neon.util.argparser import NeonArgparser
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
import os
import h5py
from matplotlib import pyplot as plt
import pickle
import matplotlib.patches as patches

#TODO log on to edison and run
# 01 is hur 10 is nhur
# parse the command line arguments
parser = NeonArgparser(__doc__)
final_dir = './results'
model_files_dir = './model_files'
images_dir = './images'
dirs = [model_files_dir, final_dir, images_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)

parser.add_argument('--load_data_from_disk')
parser.add_argument('--h5file')
parser.add_argument('--num_train')
parser.add_argument('--num_test_val')
parser.add_argument('--preproc_data_dir')
parser.set_defaults(batch_size=128,
                    test=False,
                    #save_path=model_files_dir,
                    h5file='/global/project/projectdirs/nervana/yunjie/dataset/localization_test/expand_hurricanes_loc.h5',
                    serialize=2,
                    epochs=100,
                    progress_bar=True,
                    datatype='f64',
                    model_file=False,
                    just_test=False,
                    eval_freq=1,
                    load_data_from_disk=1,
                    num_train=6,
                    num_test_val=2,
                    preproc_data_dir='/global/project/projectdirs/nervana/evan/preproc_data')

args = parser.parse_args()
args.load_data_from_disk = bool(int(args.load_data_from_disk))

num_epochs = args.epochs






args.batch_size = 100
args.eval_freq = 5


X_train, y_train, tr_i, X_test, y_test, te_i, X_val, y_val, val_i, \
nclass, \
w_size, \
cropped_ims, \
boxes= load_hurricane(path=args.h5file, num_train=int(args.num_train), num_test_val=int(args.num_test_val),
                      load_from_disk=args.load_data_from_disk, preproc_data_dir=args.preproc_data_dir)
rad = w_size / 2

print 'got data!'

train_set = DataIterator(X_train, y_train, nclass=nclass, lshape=( 8, w_size, w_size))
valid_set = DataIterator(X_val, y_val, nclass=nclass, lshape=(8, w_size, w_size))


w_init = HeWeightInit()
opt_gdm = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9)

relu = Rectlin()
conv = dict(strides=1, init=w_init, bias=Constant(0), activation=Rectlin())#, batch_norm=True)


#13 layer architecure from Ciseran et al. Mitosis Paper adjusted to fit our window size
# (they go from 101x101 as inout to 2x2 as output from last pooling layer), so we adjust to fit the output requirements
layers_13 = [Conv((2, 2, 16), **conv),
             Pooling((2, 2), strides=2),
             Conv((3, 3, 16), **conv),
             Pooling((2, 2), strides=2),
             Conv((3, 3, 16), **conv),
             Pooling((2, 2), strides=2),


             #Conv((2, 2, 16), **conv),
             #Pooling((2, 2), strides=2),
             #Conv((2, 2, 16), **conv),
             # Pooling((2, 2), strides=2),
             Affine(nout=100, init=w_init, activation=Tanh(), bias=Constant(1)),
             Affine(nout=2, init=w_init, bias=Constant(0), activation=Softmax())
]

#11 layer architecure from Ciseran et al. Mitosis Paper
layers_11 = [Conv((4, 4, 16), **conv),
             Pooling((2, 2), strides=2),
             Conv((4, 4, 16), **conv),
             Pooling((2, 2), strides=2),
             Conv((4, 4, 16), **conv),
             Pooling((2, 2), strides=2),
             Conv((3, 3, 16), **conv),
             Pooling((2, 2), strides=2),
             Affine(nout=100, init=w_init, activation=relu, bias=Constant(1)),
             Affine(nout=2, init=w_init, bias=Constant(0), activation=Softmax())
]

layers=layers_13
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers)

model_key = '{0}_{1}_{2}_{3}'.format('_'.join([((l.name[0] if l.name[0] != 'L' else 'FC') if 'Bias' not in l.name and 'Activation' not in l.name else '') +
                                                                    ('_'.join([str(num)
                                                                              for num in l.fshape]) if 'Pooling' in l.name or 'Conv' in l.name or 'Deconv' in l.name else '')
                                              for l in mlp.layers.layers]),
                                     str(args.epochs),
                                     train_set.ndata,
                                     '_'.join(map(str, train_set.shape)))

callbacks = Callbacks(mlp, train_set, args, eval_set=valid_set)
h5fin = h5py.File(os.path.join(final_dir, model_key + '-final.h5'), 'w')

#add callback for calculating train loss every every <eval_freq> epoch
callbacks.add_callback(LossCallback(h5fin.create_group('train_loss'), mlp, eval_set=train_set, epoch_freq=args.eval_freq))
callbacks.add_callback(LossCallback(h5fin.create_group('valid_loss'), mlp, eval_set=valid_set, epoch_freq=args.eval_freq))
callbacks.add_callback(MetricCallback(mlp, eval_set=train_set, metric=Misclassification(), epoch_freq=args.eval_freq))
callbacks.add_callback(MetricCallback(mlp, eval_set=valid_set, metric=Misclassification(), epoch_freq=args.eval_freq))

if args.model_file:
    mlp.load_weights(args.model_file)
else:
    mlp.fit(train_set, optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
pickle.dump(mlp.serialize(), open(os.path.join(model_files_dir, '%s.pkl' % model_key), 'w'))

print('Misclassification error = %.1f%%' % (mlp.eval(valid_set, metric=Misclassification())*100))
probs = mlp.get_outputs(valid_set)

#probs will have shape (X_val.shape[0],2) number of example_images by the output vector of 2
pos_probs = probs[:,1] #hust get second column which corresponds to prob of hurricane

#probs.reshape to n_val_im by one input image channel shape so we can have prob map
prob_map = pos_probs.reshape(val_i.shape[0], cropped_ims[0].shape[1], cropped_ims[0].shape[2])
h5fin.create_dataset('Prob_Maps', data=prob_map)
h5fin.close()

for i in range(len(val_i)):
    v_i = val_i[i]
    plt.figure(1)
    plt.clf()
    pred = plt.subplot(211)
    gr_truth = plt.subplot(212)
    pred.imshow(prob_map[i])
    gr_truth.imshow(cropped_ims[v_i, 0, :, :])
    gr_truth.add_patch(patches.Rectangle(
        (boxes[v_i][0,0] - rad,boxes[v_i][0,1] - rad),
                       boxes[v_i][0, 2] - boxes[v_i][0, 0],
                       boxes[v_i][0, 3] - boxes[v_i][0, 1],
                       fill=False))
    pred.add_patch(patches.Rectangle(
        (boxes[v_i][0,0] - rad,boxes[v_i][0,1] - rad),
                       boxes[v_i][0, 2] - boxes[v_i][0, 0],
                       boxes[v_i][0, 3] - boxes[v_i][0, 1],
                       fill=False))

    plt.savefig(os.path.join(images_dir, '%s-%i.jpg'%(os.path.splitext(os.path.basename(args.h5file))[0], i)))



