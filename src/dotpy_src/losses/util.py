


import tensorflow as tf



def add_epsilon(tens):
    epsilon = tf.constant(value=0.0001)
    return tf.add(tens,epsilon)
    

def average_nonzero_elements(tens):
    denominator = tf.count_nonzero(tens)
    denominator = tf.cast(denominator, tf.float32) + 0.001
    tens_sum = tf.reduce_sum(tens)
    mean_tens = tf.div(tens_sum, denominator)
    return mean_tens
    


def mask_tens(tens, mask):
    return tf.mul(tens, mask)   
    #tf.where(tf.equal(mask, tf.zeros_like(mask)),tf.zeros_like(tens),  tens)
    

def zero_out_nans(tens):
    return tf.where(condition=tf.is_nan(tens),x=tf.zeros_like(tens), y=tens)





