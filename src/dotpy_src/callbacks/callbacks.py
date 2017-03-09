


import keras



import sys



import numpy as np
import matplotlib; matplotlib.use('agg'); import matplotlib.pyplot as plt



class WeightPrinter(keras.callbacks.Callback):


    def on_batch_end(self, batch, logs={}):
        weights = self.model.get_weights()
        for i, weight_set in enumerate(weights):

            weights_norm = np.sum(np.square(weight_set))
            min_weight = np.min(weight_set)
            max_weight = np.max(weight_set)
            weight_shape = weight_set.shape
            sys.stderr.write("\n ####### \n Weight Set %i shape: %s\nweight norm: %5.2f \nmin_weight: %5.2f\nmax_weight: %5.2f\n ##### \n"%(i, str(weight_shape),
                                                                                                               weights_norm,
                                                                                                               min_weight,
                                                                                                               max_weight))
        sys.stderr.write(str(logs.get("loss")))
        



class LearnCurve(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.glob_losses = []

    def on_batch_end(self, batch, logs={}):
        #sys.stderr.write(str(logs))
        self.losses.append(logs.get('loss'))
    def on_epoch_end(self,epoch, logs):
        self.glob_losses.append(logs.get('loss'))
        if epoch % 1 ==0:

            plt.plot(self.glob_losses)
            pass
        
        

    def on_train_end(self,*args):
        plt.plot(self.glob_losses)
        pass



def sched(epoch_ind):
    if epoch_ind < 50:
        return 0.0001
    elif epoch_ind < 100:
        return 0.00001
    else:
        return 0.000001

lr_sched_cback = keras.callbacks.LearningRateScheduler(schedule=sched)



def get_callbacks():
    return [LearnCurve()]





