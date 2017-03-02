
# coding: utf-8

# In[3]:

import sys
from nbfinder import NotebookFinder
sys.meta_path.append(NotebookFinder())

from configs import configs
import importlib


# In[4]:

def get_loss(model_name):
    loss_name = model_name
    try:
        loss_module = importlib.import_module("notebooks.losses." + loss_name)
        loss, loss_weights = loss_module.loss, loss_module.loss_weights
        return loss, loss_weights
    except:
        loss, loss_weights = get_case_by_case_loss(model_name)
        return loss, loss_weights
        
  
        
    
        


# In[5]:

def get_case_by_case_loss(model_name):
    if model_name == "iclr_semisupervised":
        sup_loss, _ = get_loss("iclr_supervised")
        unsup_loss = "mean_squared_error"
        loss_weights = {"box_score":1, "reconstruction": configs["autoencoder_weight"]}
        loss = {"box_score":sup_loss, "reconstruction": unsup_loss}
    else:
        raise NotImplementedError
    return loss, loss_weights


# In[7]:

#! jupyter nbconvert --to script __init__.ipynb


# In[ ]:



