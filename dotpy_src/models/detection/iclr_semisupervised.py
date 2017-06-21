


import sys
import keras

from keras.models import Model

from keras.layers import Input, Conv2D,Deconv2D
if __name__ == "__main__":
    sys.path.append("../../../")
from dotpy_src.configs import configs
from dotpy_src.models.util import make_model_data_struct

from dotpy_src.models.detection.iclr_supervised import bbox
from dotpy_src.models.base.get_base_model import get_base_model_layers







layers = get_base_model_layers(name="iclr_semisupervised")
code_layer_output = layers[max([lay for lay in layers.keys() if type(lay) is not str])]



input_layer = layers[1]



box_score= bbox(code_layer_output)



reconstruction = layers["reconstruction"]



def get_model_params():
    return make_model_data_struct(input=input_layer, output = [box_score, reconstruction])








