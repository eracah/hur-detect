


import sys
import os
if os.path.basename(os.getcwd()) == "src":
    from dotpy_src.config_util import update_configs


# configs={'time_steps_per_example': 1,

#                     "time_step_sample_frequency": 2,
#                     "time_steps_per_file": 8,
#                     "image_variables" :['PRECT','PS','PSL','QREFHT','T200','T500','TMQ','TREFHT',
#                                   'TS','U850','UBOT','V850','VBOT','Z1000','Z200','ZBOT'],
#                     "label_variables" : ["x_coord","y_coord","w_coord","h_coord","obj","cls"]}

if os.path.basename(os.getcwd()) == "src":
    configs = update_configs(configs)





