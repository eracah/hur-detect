__author__ = 'racah'

import numpy as np
from data_loader import load_hurricane, is_hurricane
from neon.data import DataIterator, load_cifar10
from custom_dataiterator import CustomDataIterator
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
print sys.path
import matplotlib
from NeonMethod import NeonMethod
# 01 is hur 10 is nhur
matplotlib.use('agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches

#TODO setup having a folder per riun that all these  dirs go into like run_1_with hyperparams_calledout
# self.dirs = {'model_files_dir': './model_files/' + net_name, 'final_dir': './results/' + net_name,
              #  'images_dir': './images/' + net_name}
class Localization(NeonMethod):
    def __init__(self):
        self.args = self.setup_parser()
        super(Localization,self).__init__('fcn_localization')



    def setup_parser(self):
        parser = super(Localization,self).setup_parser()
        new_args = ['--load_data_from_disk','--num_train', '--num_test_val', '--preproc_data_dir', '--h5file', '--no_model_file', '--load_model', '--num_ims']
        for new_arg in new_args:
            parser.add_argument(new_arg)

        parser.set_defaults(batch_size=1000, test=False, serialize=2, epochs=100, progress_bar=True, datatype='f64',load_model=False, num_ims=10,
                            model_file='', load_data_from_disk=1, num_train=6, num_test_val=2, evaluation_freq=1,no_model_file=False,
                            h5file="/global/project/projectdirs/nervana/yunjie/dataset/localization_test/expand_hurricanes_loc.h5",
                            preproc_data_dir='/global/project/projectdirs/nervana/evan/preproc_data')

        args = parser.parse_args()
        args.load_data_from_disk = bool(int(args.load_data_from_disk))
        return args

    def get_data(self):


        #TODO: setup new loading of data
        data_dict = \
            load_hurricane(path=self.args.h5file,
                           num_train=int(self.args.num_train),
                           num_test_val=int(self.args.num_test_val),
                           load_from_disk=self.args.load_data_from_disk,
                           preproc_data_dir=self.args.preproc_data_dir)

        print "got data!"
        w_size = data_dict['w_size']
        nclass = data_dict['nclass']
        rad = w_size / 2
        y_val = data_dict['y_val']

        train_set = CustomDataIterator(data_dict['x_train'], y_c=data_dict['y_train'], nclass=nclass, lshape=( 8, w_size, w_size))
        valid_set = CustomDataIterator(data_dict['x_val'], y_c=data_dict['y_val'], nclass=nclass, lshape=(8, w_size, w_size))
        self.n_tr_data - train_set.ndata
        return train_set, valid_set




    def setup_network(self):
        #TODO: setup new network
        # w_init = HeWeightInit()
        # opt_gdm = GradientDescentMomentum(learning_rate=0.01, momentum_coef=0.9)
        # conv = dict(strides=1, init=w_init, bias=Constant(0), activation=Rectlin())  # , batch_norm=True)
        #
        # # 13 layer architecure from Ciseran et al. Mitosis Paper adjusted to fit our window size
        # # (they go from 101x101 as inout to 2x2 as output from last pooling layer), so we adjust to fit the output requirements
        # layers = [Conv((2, 2, 16), **conv),
        #           Pooling((2, 2), strides=2),
        #           Conv((3, 3, 16), **conv),
        #           Pooling((2, 2), strides=2),
        #           Conv((3, 3, 16), **conv),
        #           Pooling((2, 2), strides=2),
        #           Affine(nout=100, init=w_init, activation=Rectlin(), bias=Constant(1)),
        #           Affine(nout=2, init=w_init, bias=Constant(0), activation=Softmax())
        # ]
        #
        # cost = GeneralizedCost(costfunc=CrossEntropyMulti())
        # mlp = Model(layers=layers)
        #
        # model_file_path = os.path.join(self.dirs['model_files_dir'], '%s.pkl' % self.get_model_key(mlp,self.n_tr_data))
        # if os.path.exists(self.args.model_file_path) and self.args.load_model:
        #     print "loading model from file!"
        #     mlp.load_weights(self.args.model_file_path)

        return mlp, cost, opt_gdm


    def get_model_key(self, mlp, n_tr_data):

        model_key = '{0}_{1}'.format(''.join(
            [((l.name[0] if l.name[0] != 'L' else 'FC') if 'Bias' not in l.name and 'Activation' not in l.name else '') +
             ('_'.join([str(num)
                        for num in l.fshape]) if 'Pooling' in l.name or 'Conv' in l.name or 'Deconv' in l.name else '')
             for l in mlp.layers.layers]), n_tr_data)


    def save_and_plot(self, mlp):
        super(Localization,self).save_and_plot(mlp)
        self.plot_pmap()


    def plot_p_map(self):
        y_guess = self.h5fin[self.net_name + '/%s/output'%(self.eval_data_type)]
        x_input = self.h5fin['raw/%s/x'%(self.eval_data_type)]
        b_box =  self.h5fin['raw/%s/boxes'%(self.eval_data_type)]
        y = self.h5fin['raw/%s/y'%(self.eval_data_type)]

        for i in np.random.choice(x_input.shape[0],self.args.num_ims, replace=False ):
            self.save_side_by_side(x_input, y_guess, y, bbox, i)



    def save_side_by_side(self, x_in, y_guess, y, bbox, i):

        plt.figure(1)
        plt.clf()
        pred = plt.subplot(3,1,1)
        pred.imshow(y_guess)
        hur_ch = plt.subplot(3,1,2)
        hur_ch.imshow(x_in[2, :, :])
        self.add_bbox(hur_ch, bbox)
        gr_truth = plt.subplot(3,1,3)
        gr_truth.imshow(y)

        plt.savefig(os.path.join(self.dirs['images_dir'], '%s-%i.pdf' % (self.net_name, i)))

    def add_bbox(self, subplot, bbox):
        subplot.add_patch(patches.Rectangle(
        (bbox[0, 0], bbox[0, 1]),
        bbox[0, 2] - bbox[0, 0],
        bbox[0, 3] - bbox[0, 1],
        fill=False))


    def run(self):
        train_set, eval_set = self.get_data()
        mlp, cost, opt_gdm = self.setup_network()
        callbacks = self.setup_results(mlp, train_set, eval_set)
        mlp.fit(train_set,  optimizer=opt_gdm, num_epochs=self.args.epochs, cost=cost, callbacks=callbacks)
        self.evaluate(mlp, eval_set)
        #k-means here

        self.save_and_plot(mlp)
        self.h5fin.close()

    #for spearmint
    #TODO: define hyperparameters and setup spearmint
    def main(self):
        pass






#
# mlp.fit(train_set, optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
# pickle.dump(mlp.serialize(), open(model_file_path, 'w'))
#
# print('Misclassification error = %.1f%%' % (mlp.eval(valid_set, metric=Misclassification()) * 100))
#
# probs = mlp.get_outputs(valid_set)
# # probs will have shape (X_val.shape[0],2) number of example_images by the output vector of 2
# pos_probs = probs[:, 1]  # hust get second column which corresponds to prob of hurricane
# # probs.reshape to n_val_im by one input image channel shape so we can have prob map
# cropped_ims = data_dict['cropped_ims']
# val_i = data_dict['val_i']
# boxes = data_dict['boxes']
# prob_map = pos_probs.reshape(len(val_i), cropped_ims[0].shape[1], cropped_ims[0].shape[2])
# h5fin.create_dataset('Prob_Maps', data=prob_map)
# h5fin.close()




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
#TODO: Add in using just one channel

