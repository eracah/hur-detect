
# coding: utf-8

# In[1]:

import sys
import os
from nbfinder import NotebookFinder
sys.meta_path.append(NotebookFinder())


# In[2]:

import importlib


# In[13]:

import threading


# In[3]:

from configs import configs


# In[5]:

generator_module = importlib.import_module("notebooks.load_data." + configs["data_name"])


# In[10]:

def get_generator(typ, model_name):
    generator = generator_module.get_generator(typ)
    if model_name == "iclr_semisupervised":
        generator = SemisupWrapper(generator)
    return generator


# In[11]:

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


# In[16]:

#! jupyter nbconvert --to script __init__.ipynb


# In[ ]:



