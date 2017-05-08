


import os




import sys
from dotpy_src.configs import configs
from keras.models import Model
from dotpy_src.optimizers.get_opt import get_opt
from dotpy_src.losses.get_loss import get_loss
import numpy as np
from dotpy_src.models.get_model import get_model
from dotpy_src.load_data.get_generator import get_generator
from dotpy_src.fit.get_fit_function import get_fit_function



os.environ["CUDA_VISIBLE_DEVICES"]=str(configs["gpu"])



model = get_model()



opt = get_opt()

loss_func, loss_weights = get_loss(model.name)

generator = get_generator(mode=model.mode, typ="tr", )

val_generator = get_generator(typ="val", mode=model.mode)

num_epochs = configs["num_epochs"]



fit = get_fit_function()

fit(model, generator, val_generator, num_epochs, loss_func, opt)





