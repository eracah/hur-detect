


import tensorflow as tf
import sys
if __name__ == "__main__":
    sys.path.append("../../")
from dotpy_src.box_encode_decode.configs import configs
from dotpy_src.models.configs import configs as model_configs
configs.update(model_configs)



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



def unpack_net_output(pred_tensor, feat_shape, single_example=False):
    num_classes = configs["num_classes"]
    ind = configs["feat_shapes"].index(feat_shape)
    sizes, ratios = configs["anchor_sizes"][ind], configs["anchor_ratios"][ind]
    num_boxes = len(sizes)+ len(ratios)
    num_coords_per_box = 4
    loc_channels = num_boxes * num_coords_per_box
    cls_channels = num_classes * num_boxes
    """split by grabbing the first loc_channels channels
    these will be concatenated like so:
     * localizations (num_ex, ydim,xdim,loc_channels) on top of
     * logits (num_ex,ydim,xdim,cls_channels)
     so we split by the dim 3 (counting up from 0)
     then we reshape each to be: 
        * (num_ex, ydim,xdim,num_boxes, num_coords_per_box)
        * (num_ex, ydim,xdim,num_boxes, num_classes) """
    
    if single_example:
        cutoff_axis = 2
    else:
        cutoff_axis = 3
    localizations, logits = tf.split(pred_tensor, axis=cutoff_axis,num_or_size_splits=[loc_channels, cls_channels])
    #print localizations.get_shape(), logits.get_shape()
    localizations_shape = convert_tf_shape_to_int_tuple(localizations.get_shape())
    new_localization_shape = tuple(list(localizations_shape[0:cutoff_axis]) + [num_boxes, num_coords_per_box])
    localizations = tf.reshape(localizations, shape=new_localization_shape)
    
    logits_shape = convert_tf_shape_to_int_tuple(logits.get_shape())
    new_logits_shape = tuple(list(logits_shape[0:cutoff_axis]) + [num_boxes, num_classes])
    logits = tf.reshape(logits, shape=new_logits_shape)
                               
    return localizations, logits



def convert_tf_shape_to_int_tuple(tf_shape):
    return tuple([dim.value for dim in tf_shape])



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

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope


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


@add_arg_scope
def l2_normalization(
        inputs,
        scaling=False,
        scale_initializer=init_ops.ones_initializer(),
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        data_format='NHWC',
        trainable=True,
        scope=None):
    """Implement L2 normalization on every feature (i.e. spatial normalization).
    Should be extended in some near future to other dimensions, providing a more
    flexible normalization framework.
    Args:
      inputs: a 4-D tensor with dimensions [batch_size, height, width, channels].
      scaling: whether or not to add a post scaling operation along the dimensions
        which have been normalized.
      scale_initializer: An initializer for the weights.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: optional list of collections for all the variables or
        a dictionary containing a different list of collection per variable.
      outputs_collections: collection to add the outputs.
      data_format:  NHWC or NCHW data format.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      scope: Optional scope for `variable_scope`.
    Returns:
      A `Tensor` representing the output of the operation.
    """

    with variable_scope.variable_scope(
            scope, 'L2Normalization', [inputs], reuse=reuse) as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        dtype = inputs.dtype.base_dtype
        if data_format == 'NHWC':
            # norm_dim = tf.range(1, inputs_rank-1)
            norm_dim = tf.range(inputs_rank-1, inputs_rank)
            params_shape = inputs_shape[-1:]
        elif data_format == 'NCHW':
            # norm_dim = tf.range(2, inputs_rank)
            norm_dim = tf.range(1, 2)
            params_shape = (inputs_shape[1])

        # Normalize along spatial dimensions.
        outputs = nn.l2_normalize(inputs, norm_dim, epsilon=1e-12)
        # Additional scaling.
        if scaling:
            scale_collections = utils.get_variable_collections(
                variables_collections, 'scale')
            scale = variables.model_variable('gamma',
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=scale_initializer,
                                             collections=scale_collections,
                                             trainable=trainable)
            if data_format == 'NHWC':
                outputs = tf.multiply(outputs, scale)
            elif data_format == 'NCHW':
                scale = tf.expand_dims(scale, axis=-1)
                scale = tf.expand_dims(scale, axis=-1)
                outputs = tf.multiply(outputs, scale)
                # outputs = tf.transpose(outputs, perm=(0, 2, 3, 1))

        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


@add_arg_scope
def pad2d(inputs,
          pad=(0, 0),
          mode='CONSTANT',
          data_format='NHWC',
          trainable=True,
          scope=None):
    """2D Padding layer, adding a symmetric padding to H and W dimensions.
    Aims to mimic padding in Caffe and MXNet, helping the port of models to
    TensorFlow. Tries to follow the naming convention of `tf.contrib.layers`.
    Args:
      inputs: 4D input Tensor;
      pad: 2-Tuple with padding values for H and W dimensions;
      mode: Padding mode. C.f. `tf.pad`
      data_format:  NHWC or NCHW data format.
    """
    with tf.name_scope(scope, 'pad2d', [inputs]):
        # Padding shape.
        if data_format == 'NHWC':
            paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
        elif data_format == 'NCHW':
            paddings = [[0, 0], [0, 0], [pad[0], pad[0]], [pad[1], pad[1]]]
        net = tf.pad(inputs, paddings, mode=mode)
        return net


@add_arg_scope
def channel_to_last(inputs,
                    data_format='NHWC',
                    scope=None):
    """Move the channel axis to the last dimension. Allows to
    provide a single output format whatever the input data format.
    Args:
      inputs: Input Tensor;
      data_format: NHWC or NCHW.
    Return:
      Input in NHWC format.
    """
    with tf.name_scope(scope, 'channel_to_last', [inputs]):
        if data_format == 'NHWC':
            net = inputs
        elif data_format == 'NCHW':
            net = tf.transpose(inputs, perm=(0, 2, 3, 1))
        return net



if __name__=="__main__":
    feat_shape = (96,144)

    num_classes = configs["num_classes"]
    ind = configs["feat_shapes"].index(feat_shape)
    sizes, ratios = configs["anchor_sizes"][ind], configs["anchor_ratios"][ind]
    num_boxes = len(sizes)+ len(ratios)
    num_coords_per_box = 4
    loc_channels = num_boxes * num_coords_per_box
    cls_channels = num_classes * num_boxes


    pred_tensor = tf.ones( (feat_shape[0],feat_shape[1],loc_channels + cls_channels))

    pred_tensor.get_shape()

    loc, log = unpack_net_output(pred_tensor,feat_shape, single_example=True)

    #assert convert_tf_shape_to_int_tuple(loc.get_shape()) == (1,96,144,4,4)
    #assert convert_tf_shape_to_int_tuple(log.get_shape()) == (1,96,144,4,4)





