


import sys
import os
from configs import configs
import importlib
from generator.generator import GenThreadSafe, SemisupWrapper
from generator.batch_fetcher import BatchFetcher
if __name__ == "__main__":
    sys.path.append("../../")
from configs import configs
label_maker_module = importlib.import_module("dotpy_src.load_data.label_maker." + configs["label_maker_name"] + ".make_labels")



make_label_fn = label_maker_module.make_labels



def make_batch_fetcher(typ, num_examples,data_name, data_file):
    data_module = importlib.import_module("dotpy_src.load_data.datasets." + data_name+".get_data")
    ims, labels = data_module.get_data(type_=typ, data_file=data_file)
    return BatchFetcher(ims,labels,num_examples=num_examples)



def get_generator(typ, data_path=None, data_name=configs["data_name"], batch_size=None,num_ims=-1, mode="supervised" ):
    
    if data_path is None:
        data_path = configs["data_file"]
    num_ims = configs["num_"+ typ+"_ims"]
    batch_fetcher = make_batch_fetcher(num_examples=num_ims,typ=typ,
                                       data_name=data_name, data_file=data_path)
    
    if batch_size is None:
        batch_size = configs["batch_size"]
    generator = GenThreadSafe(batch_fetcher, 
                              batch_size = batch_size,
                              typ=typ, 
                              tf_mode=True, 
                              num_ex = num_ims, 
                              make_label_fxn=make_label_fn) 
    
    if mode == "semi_supervised":
        generator = SemisupWrapper(generator)
        
    return generator



gen=get_generator("tr")

