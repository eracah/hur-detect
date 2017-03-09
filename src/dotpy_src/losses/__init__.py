


import sys

from configs import configs
import importlib



def get_loss(model_name):
    loss_name = model_name
    try:
        loss_module = importlib.import_module("dotpy_src.losses." + loss_name)
        loss, loss_weights = loss_module.loss, loss_module.loss_weights
        return loss, loss_weights
    except:
        print loss_name
        loss, loss_weights = get_case_by_case_loss(loss_name)
        return loss, loss_weights
        
  
        
    
        



def get_case_by_case_loss(model_name):
    #sys.stderr.write(model_name)
    if model_name == "iclr_semisupervised":
        sup_loss, _ = get_loss("iclr_supervised")
        unsup_loss = "mean_squared_error"
        loss_weights = {"box_score":1, "reconstruction": configs["autoencoder_weight"]}
        loss = {"box_score":sup_loss, "reconstruction": unsup_loss}
    else:
        raise NotImplementedError
    return loss, loss_weights








