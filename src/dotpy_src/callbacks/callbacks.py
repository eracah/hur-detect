


import keras



import tensorflow as tf



from keras import backend as K



import sys



import numpy as np
import matplotlib; matplotlib.use('agg'); import matplotlib.pyplot as plt



import sys
from configs import configs



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



class TensorBoardLearnCurve(keras.callbacks.Callback):
    """Tensorboard basic visualizations.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    TensorBoard is a visualization tool provided with TensorFlow.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 write_graph=False,
                 write_images=False):
        super(TensorBoardLearnCurve, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        
        self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            sys.stderr.write(name + "\n")
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()



early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=configs["min_delta"], patience=configs["patience"], verbose=0, mode='auto')



TB = TensorBoardLearnCurve(log_dir='./logs/'+ configs["experiment_name"],
                                            histogram_freq=1, write_graph=False, write_images=True)



def get_callbacks():
    return [LearnCurve(), TB, early_stop]





