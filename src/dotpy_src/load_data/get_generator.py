


import sys
import os
from configs import configs
import importlib
from generator.generator import GenThreadSafe, SemisupWrapper
from generator.batch_fetcher import BatchFetcher
if __name__ == "__main__":
    sys.path.append("../../")
from configs import configs
#label_maker_module = importlib.import_module("dotpy_src.preprocessing.label_generation." + configs["label_maker_name"] + ".make_labels")



#create_labelmaker_fxn = label_maker_module.create_labelmaker_fxn



def make_batch_fetcher(typ="tr", num_examples=-1,data_name="climate", data_path=configs["data_file"]):
    data_module = importlib.import_module("dotpy_src.load_data.datasets." + data_name+".get_data")
    ims, labels = data_module.get_data(type_=typ, data_file=data_path)
    return BatchFetcher(ims,labels,num_examples=num_examples)



def get_generator(typ, data_path=configs["data_file"],
                  data_name=configs["data_name"], 
                  batch_size=None,num_ims=-1, 
                  mode="supervised" ):
    

    num_ims = configs["num_"+ typ+"_ims"]
    batch_fetcher = make_batch_fetcher(num_examples=num_ims,typ=typ,
                                       data_name=data_name, data_path=data_path)
    
    
    #all_labels = batch_fetcher.labels
   # make_label_fn = create_labelmaker_fxn(all_labels)
    
    if batch_size is None:
        batch_size = configs["batch_size"]
    generator = GenThreadSafe(batch_fetcher, 
                              shape=configs["input_shape"],#configs["input_shape"],
                              batch_size = batch_size,
                              typ=typ, 
                              tf_mode=True, 
                              num_ex = num_ims, 
                              make_label_fxn=None) 
    
    if mode == "semi_supervised":
        generator = SemisupWrapper(generator)
        
    return generator



if __name__ == "__main__":
    bf = make_batch_fetcher()

    bf.labels

    #gen=get_generator("tr")





