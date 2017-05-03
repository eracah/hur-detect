


import sys
if __name__ == "__main__":
    sys.path.append("../../../")
from dotpy_src.models.configs import configs
import importlib



def get_base_model_layers(name=None):
    base_model_name = name if name is not None else configs["base_model"]
    print base_model_name
    base_module = importlib.import_module("dotpy_src.models.base." + base_model_name)
    return base_module.get_base_layers()

