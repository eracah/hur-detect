__author__ = 'racah'
from neon.util.argparser import NeonArgparser
import numpy as np
from data_loader import load_hurricane, is_hurricane
from neon.data import DataIterator, load_cifar10
from he_initializer import HeWeightInit
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification, Tanh
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.initializers import Constant, Gaussian, GlorotUniform
from neon.models import Model
from neon.callbacks.callbacks import Callbacks, LossCallback
import os
import h5py
from matplotlib import pyplot as plt

#TODO log on to edison and run
# 01 is hur 10 is nhur
# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

args.batch_size = 1000
args.eval_freq = 5

num_epochs = 2 # args.epochs
final_dir = './results'


if not os.path.exists(final_dir):
    os.mkdir(final_dir)
path='/global/project/projectdirs/nervana/yunjie/dataset/localization/larger_hurricanes_loc.h5'
path='./raw_data/smaller_hurricanes_loc.h5'
path='./raw_data/expand_hurricanes_loc.h5'




(X_train, y_train, tr_i), (X_test, y_test, te_i), (X_val, y_val, val_i), \
nclass, \
w_size, \
cropped_ims, \
boxes= load_hurricane(path=path)
rad = w_size / 2


train_set = DataIterator(X_train, y_train, nclass=nclass, lshape=( 8, w_size, w_size))
valid_set = DataIterator(X_val, y_val, nclass=nclass, lshape=(8, w_size, w_size))


w_init = HeWeightInit()
opt_gdm = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9)

relu = Rectlin()
conv = dict(strides=1, init=w_init, bias=Constant(0), activation=Tanh())#, batch_norm=True)
#mp =
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

model_key = '{0}-{1}-{2}-{3}'.format(X_train.shape[1],'-'.join([(l.name[0] if 'Bias' not in l.name and 'Activation' not in l.name else '') +
                                                                    ('-' + str(l.fshape) if 'Pooling' in l.name or 'Conv' in l.name else '') for l in mlp.layers.layers]), str(args.epochs), str(X_train.shape[0]))

callbacks = Callbacks(mlp, train_set, args, eval_set=valid_set)
h5fin = h5py.File(os.path.join(final_dir, model_key + '-final.h5'), 'w')

#add callback for calculating train loss every every <eval_freq> epoch
callbacks.add_callback(LossCallback(h5fin.create_group('train_loss'), mlp, eval_set=train_set, epoch_freq=args.eval_freq))
callbacks.add_callback(LossCallback(h5fin.create_group('valid_loss'), mlp, eval_set=valid_set, epoch_freq=args.eval_freq))


mlp.fit(train_set,optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
print('Misclassification error = %.1f%%' % (mlp.eval(valid_set, metric=Misclassification())*100))
probs = mlp.get_outputs(valid_set)

#probs will have shape (X_val.shape[0],2) number of example_images by the output vector of 2
pos_probs = probs[:,1] #hust get second column which corresponds to prob of hurricane
#probs.reshape to n_val_im by one input image channel shape so we can have prob map
prob_map = pos_probs.reshape((orig_val_im.shape[0], orig_val_im.shape[2], orig_val_im.shape[3]))
plt.imshow(prob_map[0], interpolation='none')

gr_truth = np.zeros((orig_val_im.shape[0], orig_val_im.shape[2], orig_val_im.shape[3]))
for x in range(gr_truth.shape[0]):
    for y in range(gr_truth.shape[1]):

        b = is_hurricane(x + rad,y+rad,*boxes_te[0].T)
        gr_truth[x,y] = (1. if b else 0)
#then we need to plot X_val with bounding box