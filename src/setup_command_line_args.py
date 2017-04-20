


import importlib



import os



from os.path import join



import argparse
import sys



import json



def grab_all_configs():
    args = {}

    for dirpath, dirnames, filenames in os.walk("./dotpy_src"):
        if "ipynb_checkpoints" not in dirpath:
            package_name = dirpath.split(".")[-1].split("/")[-1]
            mod_list = [d for d in dirnames if "ipynb_checkpoints" not in d]
            if len(mod_list) > 0:
                package_name = package_name + "." if package_name is not "" else package_name

                for mod in mod_list:
                    if os.path.exists(join(dirpath, mod, "configs.py")):
                        try:

                            configs = importlib.import_module(package_name + mod + ".configs" )
                            args.update(configs.configs)
                        except:
                            pass#print "oh no " + package_name + mod + ".configs"
                    else:
                        pass
    return args



def parse_cla():
    #dump empty dict
    with open("./configs.json", "w") as f:
        json.dump({},f)
    # no notebook
    #print sys.argv[0]
    configs = {}
    if "ipykernel" not in sys.argv[0]:
        configs = _parse_cla()
        with open("./configs.json", "w") as f:
            json.dump(configs,f)
    return configs
    

    



def _parse_cla():
    configs = grab_all_configs()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for k,v in configs.iteritems():
        
        if k is not "variables":
            if type(v) is list:
                parser.add_argument('--' + k, type=type(v[0]),nargs='+', default=v, help=k)
            elif type(v) is bool:
                parser.add_argument('--' + k, action='store_true',default=v,help=k)
            else:   
                parser.add_argument('--' + k, type=type(v), default=v, help=k)

    args = parser.parse_args()
    return args.__dict__



if __name__ == "__main__":
    pass
#     cla = parse_cla()
#     for k,v in cla.iteritems():
#         print k, " : ", v





