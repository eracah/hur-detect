


import sys
if __name__ == "__main__":
    sys.path.append("../../")
from dotpy_src.config_util import update_configs

configs = {"num_epochs": 10, "fit_name": "tf_fit"}
configs = update_configs(configs)





