


import sys



from configs import configs
import importlib



from keras.models import Model



model_name = configs["model"]



model_module = importlib.import_module("dotpy_src.models." + model_name)



model_params = model_module.get_model_params()



model = Model(model_params["input"], output = model_params["output"], name=model_name)



def get_model():
    return model








