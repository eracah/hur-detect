


import sys


label_format_args = {  'num_classes': 4,
                       'box_sizes':[(64,64)],
                       'scale_factor': 64,                    
                     "variables": ['PRECT','PS','PSL','QREFHT','T200','T500','TMQ','TREFHT',
                                  'TS','U850','UBOT','V850','VBOT','Z1000','Z200','ZBOT'],
                    "xdim": 768,
                    "ydim": 1152,
                     
    
}



kwargs_list = [label_format_args ]

configs = {}
for kwargs in kwargs_list:
    configs.update(kwargs)






