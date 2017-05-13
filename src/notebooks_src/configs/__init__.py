


import sys
if __name__ == "__main__":
    sys.path.append("../../")
from os.path import dirname
import argparse



import importlib



configs = {}



config_module_names = ["box_encode_decode_configs",
                    "tensorboard_configs",
                    "fit_configs",
                    "labels_configs",
                    "load_data_configs",
                    "losses_configs",
                    "metrics_configs",
                    "models_configs",
                    "optimizers_configs"]



for config_module_name in config_module_names:
    config_module = importlib.import_module("notebooks_src.configs." + config_module_name)
    configs_dict = config_module.configs
    configs.update(configs_dict)



def _parse_cla(configs):
    
    if "ipykernel" in sys.argv[0]:
        sys.argv = sys.argv[3:] if len(sys.argv) > 3 else [sys.argv[0]]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for k,v in configs.iteritems():
        
        if k is not "variables":
            if type(v) is list:
                parser.add_argument('--' + k, type=type(v[0]),nargs='+', default=v, help=k)
            elif type(v) is bool:
                parser.add_argument('--' + k, action='store_true', default=v, help=k)
            else:   
                parser.add_argument('--' + k, type=type(v), default=v, help=k)

    args = parser.parse_args()
    configs.update(args.__dict__)
    return configs



def convert_shape_to_tf(shape):
    #shape is 3 dimensions
    new_shape = (shape[1], shape[2], shape[0])
    return new_shape



configs = _parse_cla(configs)



configs["tensor_input_shape"] = convert_shape_to_tf(configs["input_shape"])








