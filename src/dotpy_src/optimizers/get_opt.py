


#from keras.optimizers import adam,SGD



import tensorflow as tf



adam = tf.train.AdamOptimizer



SGD = tf.train.MomentumOptimizer



import sys



from configs import configs as cfg



opt_map = {"adam":adam(cfg["lr"]),
           
           
           "SGD":SGD(cfg["lr"], cfg["momentum"]) }



def get_opt():
    return opt_map[cfg["optimizer"]]
    

