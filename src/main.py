


import sys
from setup_command_line_args import parse_cla
parse_cla() 
    
    

from dotpy_src.callbacks import callbacks

from keras.models import Model
from dotpy_src.optimizers.get_opt import get_opt
from dotpy_src.losses import get_loss
import numpy as np
from dotpy_src.models import get_model
from dotpy_src.callbacks.callbacks import get_callbacks
from dotpy_src.load_data.generator import get_generator
from dotpy_src.fit.configs import configs



model = get_model()

opt = get_opt()

loss, loss_weights = get_loss(model.name)

model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights)

callbacks = get_callbacks()

generator = get_generator(mode=model.mode, typ="tr" )

val_generator = get_generator(typ="val", mode=model.mode)

model.fit_generator(generator=generator,
                    validation_data=val_generator, 
                    nb_val_samples=val_generator.num_ims,
                    samples_per_epoch=generator.num_ims,
                    callbacks=callbacks,
                    nb_epoch=configs["num_epochs"],nb_worker=4
                   )





