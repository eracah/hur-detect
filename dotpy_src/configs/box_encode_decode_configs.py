


import sys
import os 

import numpy as np





kwargs_dict = dict(    
label_configs = dict(
            feat_shapes=[(int((np.ceil(768 / float(2**i)))), 
                             int(np.ceil(1152 / float(2**i)))) for i in range(3,9) ] + [(1,1)],
            #feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12']
            anchor_size_bounds=[0.10, 0.90],
            anchor_sizes=[(20.48, 51.2),
                          (51.2, 133.12),
                          (133.12, 215.04),
                          (215.04, 296.96),
                          (296.96, 378.88),
                          (378.88, 460.8),
                          (460.8, 542.72)],
            anchor_ratios=[[2, .5],
                           [2, .5, 3, 1./3],
                           [2, .5, 3, 1./3],
                           [2, .5, 3, 1./3],
                           [2, .5, 3, 1./3],
                           [2, .5],
                           [2, .5]],
            anchor_steps=[8, 16, 32, 64, 128, 256, 512],
            anchor_offset=0.5,
            normalizations=[20, -1, -1, -1, -1, -1, -1],
            prior_scaling=[0.1, 0.1, 0.2, 0.2]),
    
)



configs = {}
for kwargs in kwargs_dict.values():
    configs.update(kwargs)





