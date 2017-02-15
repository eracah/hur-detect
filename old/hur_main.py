
import matplotlib; matplotlib.use("agg")


import sys
from scripts.train_val import *
from scripts.build_network import build_network
from scripts.load_data.netcdf_loader import BBoxIterator
from scripts.configs import *



kwargs = process_kwargs()



fns, networks = build_network(kwargs)



tv = TrainVal(BBoxIterator,kwargs, fns, networks)
if kwargs["test"] == True:
    tv.test()
else:
    tv.train()






