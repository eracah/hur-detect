


import sys
if __name__ == "__main__":
    sys.path.append("../../")
from dotpy_src.config_util import update_configs


from dotpy_src.load_data.configs import configs as data_configs



def convert_shape_to_tf(shape):
    #shape is 3 dimensions
    new_shape = (shape[1], shape[2], shape[0])
    return new_shape



configs= {"tensor_input_shape": convert_shape_to_tf(data_configs["input_shape"]),
        "num_filter_list":[128/8, 256/8, 512/8, 768/8, 1024/8],
         "base_model": "vgg16",
          "detection_model": "ssd",
          "batch_size": data_configs["batch_size"],
         "num_classes": data_configs["num_classes"], "w_decay": 0.0005, "w_init":"he_normal",
          #"feat_layers":['block4', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12']
         }






configs = update_configs(configs)





