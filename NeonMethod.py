__author__ = 'racah'

__author__ = 'racah'
import os
import pickle

from neon.data import DataIterator
from neon.layers import Conv, Pooling, GeneralizedCost, Deconv
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, SumSquared
from neon.callbacks.callbacks import Callbacks, LossCallback
from neon.util.argparser import NeonArgparser
import h5py
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import h5py
from collections import deque
from data_loaders import load_dayabay_conv
from custom_initializers import HeWeightInit
import abc
import numpy as np
matplotlib.use('agg')


class NeonMethod(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, net_name):
        self.net_name = net_name
        self.dirs = {'model_files_dir': './model_files/' + net_name, 'final_dir': './results/' + net_name,
                'images_dir': './images/' + net_name}
        self.setup_dirs(**self.dirs)
        self.eval_data_type = 'test' if self.args.test else 'val'
        self.h5fin, self.final_h5_filename = self.create_h5_file(self.final_dir,
                                                            self.args.epochs, self.args.learn_rate)

    def setup_parser(self):
        # parse the command line arguments
        parser = NeonArgparser(__doc__)

        parser.add_argument('--h5file')
        parser.add_argument('--test')
        parser.add_argument('--learn_rate')
        # parser.add_argument('--wrap_pad_trick')
        # parser.add_argument('--cylinder_local_trick')
        # parser.add_argument('--bneck_width')
        # parser.add_argument('--max_tsne_iter')
        # parser.set_defaults(batch_size=100,h5file='/global/homes/p/pjsadows/data/dayabay/single/single_20000.h5',
        #             serialize=2, epochs=100, learn_rate=0.0001, model_file=False,eval_freq=1, test=False, save_path=self.model_files_dir,
        #             wrap_pad_trick=False, cylinder_local_trick=False, bneck_width=10, max_tsne_iter=500)
        #
        # args = parser.parse_args()
        # args.learn_rate = float(args.learn_rate)
        # args.max_tsne_iter = int(args.max_tsne_iter)
        # return args
        return parser




    def setup_dirs(self,**kwargs):
        for dir in kwargs.values():
            if not os.path.exists(dir):
                os.makedirs(dir)

    @abc.abstractmethod
    def get_data(self):
        return
        # (X_train, y_train), (X_val,y_val), \
        # (X_test, y_test), nclass = load_dayabay_conv(path=self.args.h5file,
        #                                             clev_preproc=False, seed=6, eq_class=True, get_y=True )
        #
        # X_train, y_train, X_val, y_val,\
        # X_test, y_test = adjust_train_val_test_sizes(self.args.batch_size, X_train,
        #                                              y_train, X_val, y_val, X_test, y_test)
        #
        #
        # save_orig_data(self.h5fin, X_train, y_train, X_val, y_val, X_test, y_test)
        # self.num_tr_examples = X_train.shape[0]
        # print self.num_tr_examples
        # self.data_dim = X_train.shape[1]
        #
        #
        # train_set = DataIterator(X_train, lshape=(1, 8, 24), make_onehot=False)
        # valid_set = DataIterator(X_val, lshape=(1, 8, 24), make_onehot=False)
        #
        # if self.args.test:
        #     train_set = DataIterator(np.vstack((X_train, X_val)), lshape=(1, 8, 24), make_onehot=False)
        #     test_set = DataIterator(X_test, lshape=(1, 8, 24), make_onehot=False)
        #     return train_set, test_set
        # else:
        #     return train_set, valid_set

    @abc.abstractmethod
    def setup_network(self):
        pass
        #
        # w_init = HeWeightInit()
        #
        # opt_gdm = GradientDescentMomentum(learning_rate=self.args.learn_rate, momentum_coef=0.9)
        #
        # conv = dict(strides=1, init=w_init, padding={'pad_w': 0, 'pad_h':1}, activation=Rectlin(),
        #             batch_norm=False)#, batch_norm=True)
        # dconv = dict(init=w_init, strides=2, padding=0, batch_norm=False)
        #
        # #change kernal size to 5x5
        # if self.args.wrap_pad_trick: #doesn't work until newer version of neon 1.1.1 and above I think
        #     filter_size = 3
        #     def transform(X):
        #         X = X.reshape(X.shape[0],8,24)
        #         #pad the right with the first filter_size-1 columns from the left
        #         X_p = np.lib.pad(X, ((0, 0), (0, 0), (0, filter_size - 1)), 'wrap')
        #         X_p = X_p.reshape(X.shape[0],8*26)
        #         return X_p
        #     layers = [DataTransform(transform=transform),Conv((3, 3, 16), **conv)] #8,26,2 -> 8,24,]
        # else:
        #     layers = [Conv((5, 5, 16), strides=1, init=w_init, padding=2, activation=Rectlin(), batch_norm=False)] #8,24,1-> 8,24,
        #
        # layers.extend([
        #           Pooling((2, 2), strides=2),# -> 4,12,
        #           Conv((3, 3, 16), **conv), # -> 4,10,
        #           Pooling((2, 2), strides=2), #-> 2,5
        #           Conv((2, 5, self.args.bneck_width), init=w_init, strides=1, padding=0, activation=Rectlin(),batch_norm=False, act_name="middleLayer"),#-> 1,1,10 like an FC layer
        #           Deconv((2, 4, 16), **dconv), #-> 2,4,
        #           Deconv((2, 5, 16), init=w_init, strides=2, padding=0, batch_norm=False), #-> 4,11
        #           Deconv((2, 4, 1), **dconv)] )#->8,24,
        #
        #
        # # Define the cost
        # cost = GeneralizedCost(costfunc=SumSquared())
        # mlp = Model(layers=layers)
        # return mlp, cost, opt_gdm

    def get_model_key(self, mlp, n_tr_data):
        return str(mlp)
        # model_key = '{0}-{1}-{2}-{3}-{4}'.format(self.data_dim,'-'.join([(l.name[0] if 'Bias' not in l.name and 'Activation' not in l.name else '') +
        # ('-' + '_'.join(str(l.fshape).split(' '))
        # if 'Pooling' in l.name or 'Conv' in l.name or 'conv' in l.name else '')
        #                                                       for l in mlp.layers.layers]),
        #                               str(self.args.epochs),
        #                               str(self.num_tr_examples),
        #                               str(self.data_dim))
        # return model_key


    def save_orig_data(self,h5fin,X_train, y_train, X_val,y_val, X_test, y_test):
        h5fin.create_dataset('raw/train/x', data=X_train)
        h5fin.create_dataset('raw/train/y', data=y_train)
        h5fin.create_dataset('raw/test/x', data=X_test)
        h5fin.create_dataset('raw/test/y', data=y_test)
        h5fin.create_dataset('raw/val/x', data=X_val)
        h5fin.create_dataset('raw/val/y', data=y_val)


    def plot_train_val_learning_curve(self,final_h5_file):
        h5fin = h5py.File(final_h5_file)
        plt.clf()
        v_loss = np.asarray(h5fin['valid_loss/cost/loss'])
        t_loss = np.asarray(h5fin['train_loss/cost/loss'])
        plt.plot(range(v_loss.shape[0]), v_loss)
        plt.plot(range(t_loss.shape[0]), t_loss)
        plt.xlabel('epochs')
        plt.ylabel('sum squared error')
        plt.legend(['val_loss', 'tr_loss'])
        plt.savefig(final_h5_file + '.pdf')


    def adjust_train_val_test_sizes(self,batch_size, X_train, y_train, X_val, y_val, X_test, y_test ):
        #make sure size of validation data a multiple of batch size (otherwise tough to match labels)

        train_end = batch_size * (X_train.shape[0] / batch_size)
        X_train = X_train[:train_end]
        y_train = y_train[:train_end]

        val_end = batch_size * (X_val.shape[0] / batch_size)
        X_val = X_val[:val_end]
        y_val = y_val[:val_end]

        #make sure size of test data a multiple of batch size
        test_end = batch_size * (X_test.shape[0] / batch_size)
        X_test = X_test[:test_end]
        y_test = y_test[:test_end]

        return X_train, y_train, X_val, y_val, X_test, y_test



    def create_h5_file(self,final_dir, *args):
        final_h5_file = os.path.join(final_dir, 'final_results_tr_size' + '_'.join(map(lambda x: str(x), args)) + '.h5')
        h5fin = h5py.File(final_h5_file, 'w')
        return h5fin, final_h5_file

    def setup_results(self, mlp, train_set, eval_set):

        model_key = self.get_model_key(mlp,train_set)
        self.args.save_path = self.model_files_dir + '/' + model_key + '.pkl'




        callbacks = Callbacks(mlp, train_set, self.args) # eval_set=eval_set the two callbacks below do this
        callbacks.add_save_best_state_callback(self.model_files_dir)

        callbacks.add_callback(LossCallback(( self.h5fin.get('train_loss', False) if 'train_loss' in self.h5fin else self.h5fin.create_group('train_loss')),
                                            mlp, eval_set=train_set, epoch_freq=self.args.eval_freq))
        if not self.args.test:
            callbacks.add_callback(LossCallback(( self.h5fin.get('valid_loss', False) if 'valid_loss' in self.h5fin else self.h5fin.create_group('valid_loss')),
                                                mlp, eval_set=eval_set, epoch_freq=self.args.eval_freq))
            #callbacks.add_early_stop_callback(stop_func)
        else:
            callbacks.add_callback(LossCallback(( self.h5fin.get('test_loss', False) if 'test_loss' in self.h5fin else self.h5fin.create_group('valid_loss')),
                                                mlp, eval_set=eval_set, epoch_freq=self.args.eval_freq))

        return callbacks




    def evaluate(self, mlp, eval_set):


        h5ae_eval = self.h5fin.create_dataset(self.net_name + '/%s/x' % (self.eval_data_type),
                                              (eval_set.ndata,
                                               self.args.bneck_width)) if  self.net_name + '/%s/x'%(self.eval_data_type) not in self.h5fin else self.h5fin[self.net_name + '/%s/x' %(self.eval_data_type)]
        # save_middle_layer_output(eval_set, h5ae_eval, mlp, self.args.bneck_width)
        eval_output = mlp.get_outputs(eval_set)
        self.h5fin.create_dataset(self.net_name + '/%s/output'%(self.eval_data_type), data=eval_output)


    def save_and_plot(self, mlp):

        self.plot_train_val_learning_curve(self.final_h5_filename)



        # v = Vis(self.final_h5_filename, old=False, plot_tsne=True, reconstruct=False, pp_types='conv-ae,raw', data_types=self.eval_data_type, max_iter=self.args.max_tsne_iter)
        # v.plot()

        pickle.dump(mlp.serialize(), open(os.path.join(self.model_files_dir, '%s-%s-%s.pkl'%(self.model_key, str(self.args.epochs), str(self.args.learn_rate))), 'w'))

    def main(self):
        train_set, eval_set = self.get_data()

        mlp, cost, opt_gdm = self.setup_network()
        callbacks = self.setup_results(mlp, train_set, eval_set)
        mlp.fit(train_set,  optimizer=opt_gdm, num_epochs=self.args.epochs, cost=cost, callbacks=callbacks)
        self.evaluate(mlp, eval_set)
        #k-means here
        self.h5fin.close()
        self.save_and_plot(mlp)


