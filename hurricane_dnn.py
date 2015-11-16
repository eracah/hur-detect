__author__ = 'racah'
from neon.util.argparser import NeonArgparser
import numpy as np
from data_loader import load_hurricane
from neon.data import DataIterator, load_cifar10
from he_initializer import HeWeightInit
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.initializers import Constant, Gaussian, GlorotUniform
from neon.models import Model
from neon.callbacks.callbacks import Callbacks


# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()
args.batch_size = 256
# 01 is hur
# hyperparameters
num_epochs = 1 # args.epochs
path='/global/project/projectdirs/nervana/yunjie/dataset/localization/larger_hurricanes_loc.h5'
path='./raw_data/smaller_hurricanes_loc.h5'
(X_train, y_train, orig_train_im),\
(X_test, y_test, orig_test_im),\
(X_val, y_val, orig_val_im),\
nclass, \
w_size= load_hurricane(path=path)


train_set = DataIterator(X_train, y_train, nclass=nclass, lshape=( 8, w_size, w_size))
valid_set = DataIterator(X_val, y_val, nclass=nclass, lshape=(8, w_size, w_size))


w_init = HeWeightInit()
opt_gdm = GradientDescentMomentum(learning_rate=0.001, momentum_coef=0.9)

relu = Rectlin()
conv = dict(strides=1, init=w_init, bias=Constant(0), activation=relu)
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
             Affine(nout=100, init=w_init, activation=relu, bias=Constant(1)),
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

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers_13)

callbacks = Callbacks(mlp, train_set, args, eval_set=valid_set)

mlp.fit(train_set,optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
print('Misclassification error = %.1f%%' % (mlp.eval(valid_set, metric=Misclassification())*100))
probs = mlp.get_outputs(valid_set)

#probs will have shape (X_val.shape[0],2) number of example_images by the output vector of 2
pos_probs = probs[:,0] #hust get first column which corresponds to prob of hurricane
#probs.reshape to n_val_im by one input image channel shape so we can have prob map
probs.reshape((orig_val_im.shape[0], X_val.shape[2], X_val.shape[3]))
#then we need to plot X_val with bounding box