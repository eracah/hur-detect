
# coding: utf-8

# In[2]:

# import sys
# from nbfinder import NotebookFinder
# sys.meta_path.append(NotebookFinder())


# In[1]:

from configs import configs
import importlib


# In[2]:

from keras.models import Model


# In[3]:

model_name = configs["model"]


# In[4]:

model_module = importlib.import_module("notebooks.models." + model_name)


# In[5]:

model_params = model_module.get_model_params()


# In[6]:

model = Model(model_params["input"], output = model_params["output"], name=model_name)


# In[7]:

def get_model():
    return model


# In[9]:

#! jupyter nbconvert __init__.ipynb --to script


# In[ ]:



