


import sys
import os



import importlib



import threading



from configs import configs



generator_module = importlib.import_module("dotpy_src.load_data." + configs["data_name"])



def get_generator(typ, model_name):
    generator = generator_module.get_generator(typ)
    if model_name == "iclr_semisupervised":
        generator = SemisupWrapper(generator)
    return generator



#thread safe
class SemisupWrapper(object):
    def __init__(self,generator):
        self.generator = generator
        self.lock = threading.Lock()
        
    def __iter__(self):
        return self
    
    @property
    def num_ims(self):
        return self.generator.num_ims
    
    def next(self):
        with self.lock:
            ims, lbls = self.generator.next()
            return ims, {"box_score":lbls,"reconstruction":ims}








