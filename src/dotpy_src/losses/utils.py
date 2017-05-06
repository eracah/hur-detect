


import tensorflow as tf
import sys
if __name__ == "__main__":
    sys.path.append("../../")
from dotpy_src.configs import configs



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



# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implement some custom layers, not provided by TensorFlow.
Trying to follow as much as possible the style/standards used in
tf.contrib.layers
"""
import tensorflow as tf

def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.
    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r





