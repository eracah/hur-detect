


import sys
import keras


#from keras.initializations import he_normal

from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Conv2D,merge, GlobalAveragePooling2D, Dense, Reshape, Activation
from keras.layers.merge import concatenate
import importlib
if __name__ == "__main__":
    sys.path.append("../../../")
from dotpy_src.configs import configs
from dotpy_src.models.util import make_model_data_struct, Normalize
from dotpy_src.models.base.get_base_model import get_base_model_layers  




# from IPython.display import Image
# Image(filename='./../ssd-resnet.png') 

# Image(filename="./../ssd-vgg.png")



conv_kwargs =  dict(padding="same",kernel_initializer=configs["w_init"])



def add_ssd_additional_feat_layers(layers):

    layers['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(layers['conv4_3'])
    
    layers['fc6'] = Conv2D(1024, (3, 3), dilation_rate=6,
                                     activation='relu', padding='same',
                                     name='fc6')(layers['pool5'])
    # FC7
    layers['fc7'] = Conv2D(1024,(1, 1), activation='relu',
                               padding='same', name='fc7')(layers['fc6'])
    # x = Dropout(0.5, name='drop7')(x)
    # Block 6
    layers['conv8_1'] = Conv2D(256, (1, 1), activation='relu',
                                   padding='same',
                                   name='conv8_1')(layers['fc7'])
    layers['conv8_2'] = Conv2D(512, (3, 3), strides=(2, 2),
                                   activation='relu', padding='same',
                                   name='conv8_2')(layers['conv8_1'])
    # Block 7
    layers['conv9_1'] = Conv2D(128, (1, 1), activation='relu',
                                   padding='same',
                                   name='conv9_1')(layers['conv8_2'])
    #layers['conv9_2'] = ZeroPadding2D()(layers['conv9_1'])
    layers['conv9_2'] = Conv2D(256, (3, 3), strides=(2, 2),
                                   activation='relu', padding='same',
                                   name='conv9_2')(layers['conv9_1'])
    # Block 8
    layers['conv10_1'] = Conv2D(128, (1, 1), activation='relu',
                                   padding='same',
                                   name='conv10_1')(layers['conv9_2'])
    layers['conv10_2'] = Conv2D(256, (3, 3), strides=(2, 2),
                                   activation='relu', padding='same',
                                   name='conv10_2')(layers['conv10_1'])
    
    
    
    layers['conv11_1'] = Conv2D(128, (1, 1), activation='relu',
                                   padding='same',
                                   name='conv11_1')(layers['conv10_2'])
    layers['conv11_2'] = Conv2D(256, (3, 3), strides=(2, 2),
                                   activation='relu', padding='same',
                                   name='conv11_2')(layers['conv11_1'])
    # Last Pool
    layers['pool12'] = GlobalAveragePooling2D(name='pool6')(layers['conv11_2'])

    return layers
    
    
    
    



def get_ssd_detection_outputs(layers):
    num_anchors = [len(sizes) + len(ratios) for sizes, ratios in zip(configs["anchor_sizes"],configs["anchor_ratios"])]
    net_out_dict={}
    num_classes = configs["num_classes"]
    
    layer_names_for_pred = ['conv4_3_norm','fc7','conv8_2', 'conv9_2','conv10_2', 'conv11_2', 'pool12' ]
    for ind, name in enumerate(layer_names_for_pred):
        fmap = layers[name]
        layers[name + "_xywh_output"], layers[name + "_cls_output"] = get_detections_for_fmap(name, fmap, num_anchors[ind])
        layers[name + "_combined_output"] = concatenate([layers[name + "_xywh_output"],
                                                         layers[name + "_cls_output"]],axis=3)

        

    return layers # [lay for lay in layers if "combined" in lay]



def get_detections_for_fmap(name, fmap, num_anchors):
        num_classes = configs["num_classes"]
        num_xywh_outputs = num_anchors * 4
        num_cls_outputs = num_anchors * num_classes
        if "pool" in name:
            xywh_output = Dense(num_xywh_outputs)(fmap)
            xywh_output = Reshape(target_shape=(1,1, num_xywh_outputs))(xywh_output)
            cls_output = Dense(num_cls_outputs)(fmap)
            cls_output = Reshape((1,1,num_cls_outputs))(cls_output)
            

        else:
            xywh_output = Conv2D( num_xywh_outputs, (3, 3), activation="linear", **conv_kwargs)(fmap)

            #no softmax for now?
            cls_output = Conv2D(num_cls_outputs, (3, 3), activation="linear", **conv_kwargs)(fmap)
            
        
        
        return xywh_output, cls_output
        
    



# # layers is a dict matching local receptive field to layer
#pseudoish-code
layers = get_base_model_layers()
layers = add_ssd_additional_feat_layers(layers)
layers = get_ssd_detection_outputs(layers)



outputs = [layers[k] for k in sorted(layers.keys()) if "combined" in k]



input_tensor = layers["input"]



def get_model_params():
    return make_model_data_struct(inputs=input_tensor, outputs=outputs)

