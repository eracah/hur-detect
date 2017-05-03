



"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional layersworks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
import sys
import keras
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D,Flatten, ZeroPadding2D, AveragePooling2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Input

if __name__ == "__main__":
    sys.path.append("../../../") 
from dotpy_src.models.configs import configs
from dotpy_src.load_data.configs import configs as data_configs
configs.update(data_configs)

#WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG16(inp_shape, batch_size=None):


    layers = {}
    if batch_size is None:
        img_input = Input(shape=inp_shape)
    else:
        inp_shape = tuple([batch_size] + list(inp_shape))
        img_input =Input(batch_shape=inp_shape)
    layers['input'] = img_input
    layers['conv1_1'] = Conv2D(64, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv1_1')(layers['input'])
    layers['conv1_2'] = Conv2D(64, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv1_2')(layers['conv1_1'])
    layers['pool1'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool1')(layers['conv1_2'])
    # Block 2
    layers['conv2_1'] = Conv2D(128, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv2_1')(layers['pool1'])
    layers['conv2_2'] = Conv2D(128, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv2_2')(layers['conv2_1'])
    layers['pool2'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool2')(layers['conv2_2'])
    # Block 3
    layers['conv3_1'] = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_1')(layers['pool2'])
    layers['conv3_2'] = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_2')(layers['conv3_1'])
    layers['conv3_3'] = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_3')(layers['conv3_2'])
    layers['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool3')(layers['conv3_3'])
    # Block 4
    layers['conv4_1'] = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv4_1')(layers['pool3'])
    layers['conv4_2'] = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv4_2')(layers['conv4_1'])
    layers['conv4_3'] = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv4_3')(layers['conv4_2'])
    layers['pool4'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool4')(layers['conv4_3'])
    # Block 5
    layers['conv5_1'] = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_1')(layers['pool4'])
    layers['conv5_2'] = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_2')(layers['conv5_1'])
    layers['conv5_3'] = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_3')(layers['conv5_2'])
    layers['pool5'] = MaxPooling2D((3, 3), strides=(1, 1), padding='same',
                                name='pool5')(layers['conv5_3'])
    return layers
    



def get_base_layers(inp_shape=None, batch_size=None):
    if inp_shape is None:
        inp_shape = configs["tensor_input_shape"]

    if batch_size is None:
        batch_size = configs["batch_size"]
    layers_dict = VGG16(inp_shape, batch_size)
    return layers_dict





