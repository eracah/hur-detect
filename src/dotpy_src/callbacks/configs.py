


import sys
import os 
if os.path.basename(os.getcwd()) == "src":
    from dotpy_src.config_util import update_configs



runs = os.listdir("./logs")



num = len(runs)



configs = {"experiment_name": "run_" + str(num), "min_delta":0.001, "patience":5}



if os.path.basename(os.getcwd()) == "src":
    configs = update_configs(configs)

