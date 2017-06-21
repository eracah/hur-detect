


import sys
if __name__ == "__main__":
    sys.path.append("../../")
from dotpy_src.configs import configs



import importlib



def get_fit_function():
    fit_module = importlib.import_module("dotpy_src.fit." + configs["fit_name"])
    fit_fxn = fit_module.fit
    return fit_fxn
    

