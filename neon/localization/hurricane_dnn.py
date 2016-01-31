__author__ = 'racah'

import numpy as np
from data_loader import LoadHurricane
from neon.data import DataIterator, load_cifar10
from custom_dataiterator import CustomDataIterator
from custom_initializers import HeWeightInit, AveragingFilterInit
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification, Tanh, SumSquared, CrossEntropyBinary, Logistic
from neon.layers import Conv, Deconv, Dropout, Pooling, GeneralizedCost, Affine
from neon.initializers import Constant, Gaussian, GlorotUniform
from neon.models import Model
from neon.callbacks.callbacks import Callbacks, LossCallback, MetricCallback
from neon.util.argparser import NeonArgparser
from custom_cost import MeanCrossEntropyBinary
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
import time


class Localization(NeonMethod):
    def __init__(self, run_results_dir):
        self.args = self.setup_parser()
        super(Localization,self).__init__('fcn_localization', run_results_dir)




    def setup_parser(self):
        parser = super(Localization,self).setup_parser()
        new_args = ['--load_data_from_disk','--num_train', '--num_test_val',
                    '--preproc_data_dir', '--load_model', '--num_ims', '--ims_to_plot']
        for new_arg in new_args:
            parser.add_argument(new_arg)

        parser.set_defaults(batch_size=1000, test=False, serialize=2, epochs=100, progress_bar=True, datatype='f64',load_model=0, num_ims=None,
         load_data_from_disk=1, eval_freq=1,ims_to_plot=5,
                            h5file="/global/project/projectdirs/nervana/yunjie/dataset/localization_test/expand_hurricanes_loc.h5",
                            preproc_data_dir='/global/project/projectdirs/nervana/evan/preproc_data')

        args = parser.parse_args()
        args.load_data_from_disk = bool(int(args.load_data_from_disk))
        args.load_model = bool(int(args.load_model))
        args.num_ims = int(args.num_ims)
        print args.batch_size
        return args

    def get_data(self):
        lh = LoadHurricane(flatten=True, num_ims=self.args.num_ims, batch_size=self.args.batch_size)
        (X_tr, y_tr, bbox_tr), \
        (X_te, y_te, bbox_te), \
        (X_val, y_val, bbox_val), (x_dims, y_dims) = lh.load_hurricane(path=self.args.h5file).values()

        self.save_orig_data(self.h5fin, X_tr, y_tr, X_val, y_val, X_te, y_te, bbox_tr, bbox_te, bbox_val)
        print X_tr.shape[1:]
        self.x_dims = x_dims
        self.y_dims = y_dims
        train_set = CustomDataIterator(X_tr, lshape=x_dims,y_c=y_tr,)
        valid_set = CustomDataIterator(X_val, y_c=y_val,lshape=x_dims)
        self.n_tr_data = train_set.ndata


        return train_set, valid_set




    def setup_network(self):
        #TODO: have network output 7x7 region (shorter train time)
        w_init = HeWeightInit()
        opt_gdm = GradientDescentMomentum(learning_rate=0.001, momentum_coef=0.9)
        conv = dict(strides=1, init=w_init, activation=Rectlin())  # , batch_norm=True)
        dconv = dict(init=w_init, strides=1, padding=0, batch_norm=False)
        #TODO code up simple architecture
        # layers = 5*[Conv((2, 2, 16), **conv)] + 4*[Deconv((2,2,16), **dconv)] +\
        #  [Deconv((2, 2,2),init=w_init, strides=1, padding=0, batch_norm=False, activation= Softmax() )]
        layers = [Conv((2, 2, 16), **conv), Deconv((2, 2,2),init=w_init, strides=1, padding=0, batch_norm=False, activation= Softmax() )]


        cost = GeneralizedCost(costfunc=MeanCrossEntropyBinary())
        mlp = Model(layers=layers)
        model_file_path = os.path.join(self.dirs['prev_model_dir'],self.get_model_key(mlp,self.n_tr_data))
        if os.path.exists(model_file_path + '.pkl') and self.args.load_model:
            print "loading model from file!"
            mlp.load_weights(model_file_path + '.pkl')

        return mlp, cost, opt_gdm


    def get_model_key(self, mlp, n_tr_data):

        model_key = '{0}_{1}'.format(''.join(
            [((l.name[0] if l.name[0] != 'L' else 'FC') if 'Bias' not in l.name and 'Activation' not in l.name else '') +
             ('_'.join([str(num)
                        for num in l.fshape]) if 'Pooling' in l.name or 'Conv' in l.name or 'Deconv' in l.name else '')
             for l in mlp.layers.layers]), n_tr_data)
        return model_key

    def save_and_plot(self, mlp, n_tr_data):
        super(Localization,self).save_and_plot(mlp, n_tr_data)
        self.plot_p_map()


    def plot_p_map(self):
        y_guess = np.asarray(self.h5fin[self.net_name + '/%s/output'%(self.eval_data_type)])
        x_input = np.asarray(self.h5fin['raw/%s/x'%(self.eval_data_type)])
        b_box =  np.asarray(self.h5fin['raw/%s/boxes'%(self.eval_data_type)])
        y = np.asarray(self.h5fin['raw/%s/y'%(self.eval_data_type)])

        for i in np.random.choice(x_input.shape[0],self.args.ims_to_plot, replace=False ):
            self.save_side_by_side(x_input, y_guess, y, b_box, i)



    def save_side_by_side(self, x_in, y_guess, y, bbox, i):

        x_in = x_in.reshape(x_in.shape[0],*self.x_dims)
        print self.y_dims
        y = y.reshape(y.shape[0], *self.y_dims)
        y_guess = y_guess.reshape(y_guess.shape[0], *self.y_dims)
        plt.figure(1)
        plt.clf()
        pred = plt.subplot(3,1,1)
        pred.imshow(y_guess[i,0])
        hur_ch = plt.subplot(3,1,2)
        hur_ch.imshow(x_in[i,3, :, :])
        self.add_bbox(hur_ch, bbox)
        gr_truth = plt.subplot(3,1,3)
        gr_truth.imshow(y[i,0])

        plt.savefig(os.path.join(self.dirs['images_dir'], '%s-%i.pdf' % (self.net_name, i)))

    def add_bbox(self, subplot, bbox):
        subplot.add_patch(patches.Rectangle(
        xy=(bbox[0], bbox[1]),
        width=bbox[2] - bbox[0],
        height=bbox[3] - bbox[1],
        fill=False))


    def run(self):

        train_set, eval_set = self.get_data()
        mlp, cost, opt_gdm = self.setup_network()
        callbacks = self.setup_results(mlp, train_set, eval_set)
        t0 = time.time()
        mlp.fit(train_set,  optimizer=opt_gdm, num_epochs=self.args.epochs, cost=cost, callbacks=callbacks)
        t1 = time.time()
        tr_time = t1 - t0
        self.evaluate(mlp, eval_set)
        print "train time: %i", tr_time

        self.save_and_plot(mlp, self.n_tr_data)
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


#
#
# # plot
# for i in range(len(val_i)):
#     v_i = val_i[i]
#     plt.figure(1)
#     plt.clf()
#     pred = plt.subplot(3,1,1)
#     pred.imshow(prob_map[i])
#     hur_ch = plt.subplot(3,1,2)
#     hur_ch.imshow(cropped_ims[v_i, 2, :, :])
#     hur_ch.add_patch(patches.Rectangle(
#         (boxes[v_i][0, 0] - rad, boxes[v_i][0, 1] - rad),
#         boxes[v_i][0, 2] - boxes[v_i][0, 0],
#         boxes[v_i][0, 3] - boxes[v_i][0, 1],
#         fill=False))
#     # pred.add_patch(patches.Rectangle(
#     #     (boxes[v_i][0, 0] - rad, boxes[v_i][0, 1] - rad),
#     #     boxes[v_i][0, 2] - boxes[v_i][0, 0],
#     #     boxes[v_i][0, 3] - boxes[v_i][0, 1],
#     #     fill=False))
#     gr_truth = plt.subplot(3,1,3)
#     x=cropped_ims[0].shape[1]
#     y=cropped_ims[0].shape[2]
#     gr_truth.imshow(y_val[i*x*y : (i+1)*x*y].reshape(x,y))
#
#     plt.savefig(os.path.join(dirs['images_dir'], '%s-%i.pdf' % (model_key, i)))
#

#TODO: Add in learning curve
#TODO: Add in using just one channel

if __name__ == "__main__":
    l = Localization('./run_results')
    l.run()
