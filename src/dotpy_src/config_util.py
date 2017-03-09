


import json
import os
from setup_command_line_args import parse_cla



def update_configs(configs):
    cl_kwargs = parse_cla()
    
    for k,v in cl_kwargs.iteritems():
        if k in configs:
            configs[k] = v
    return configs

