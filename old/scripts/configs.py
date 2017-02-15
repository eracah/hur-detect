
import matplotlib; matplotlib.use("agg")


import sys
from helper_fxns import *
from run_dir import *
from lasagne.nonlinearities import *
from lasagne.init import *
import argparse



all_args = dict(data_format_args = {'input_shape': (None,16,768,1152),
                    '3d_time_steps_per_example': 8,
                    "im_dim": 3,
                    "variables": ['PRECT','PS','PSL','QREFHT','T200','T500','TMQ','TREFHT',
                                  'TS','U850','UBOT','V850','VBOT','Z1000','Z200','ZBOT'],
                    "xdim": 768,
                    "ydim": 1152,
                    "time_step_sample_frequency": 2,
                    "time_steps_per_file": 8
    
},


                
                

label_format_args = {  'num_classes': 4,
                       'box_sizes':[(64,64)],
                       'scale_factor': 64,
                     
    
},


tr_val_test_args = {'batch_size' : 1,
                    'num_test_days':365,
                    'num_tr_days': 365,
                    'tr_years': [1979],
                    'val_years': [1980],
                    "test_years" : [1984],
                    'shuffle': False,
                    'epochs': 10000,
                    "test": False,
                   },

file_args = {'metadata_dir': "/home/evan/data/climate/labels/",
             'data_dir': "/home/evan/data/climate/input",

             "max_files_open": 1,
             
    
},


opt_args = { 'learning_rate': 0.0001,
             'weight_decay': 0.0005, 
             'lambda_ae' : 10,          
             'coord_penalty': 5,
             'size_penalty': 7,
             'nonobj_penalty': 0.5,
             "batch_norm" : False,
             'dropout_p': 0, 
             "yolo_batch_norm" : True,
    
},

arch_args = {
             "filters_scale" : 1.,
            "filter_dim" : 5, "num_layers": 6

},


eval_args = {
    'iou_thresh' : 0.5,

    
},


save_load_args = { 'save_weights': True,
                   "yolo_load_path": "None",
                   "ae_load_path": "None",
                   "save_path":"./results",
                  "save_results": True,
},


plotting_args = {  "num_ims_to_plot" : 8,

                  "get_fmaps": False,



                  "no_plots": False,

                  "get_ims": False
                },
    )


default_args = {}
[default_args.update(d) for d in all_args.values() ]



def process_kwargs(save_res=True):
    args= parse_cla()
    kwargs = default_args
    
    kwargs.update(args)


    if kwargs["save_results"]:
        run_dir = create_run_dir(save_path)
        kwargs['save_path'] = run_dir
        dump_hyperparams(kwargs, run_dir)

        kwargs["logger"] = setup_logging(kwargs['save_path'])
    
    return kwargs



def parse_cla():
    # if inside a notebook, then get rid of weird notebook arguments, so that arg parsing still works
    if any(["jupyter" in arg for arg in sys.argv]):
        sys.argv=sys.argv[:1]
        #default_args.update({"num_layers": 6, "num_test_days":3,"ignore_plot_fails":0, "test":False, "no_plots":True, "num_filters": 2, "filters_scale": 0.01, "num_tr_days":3, "lambda_ae":0})


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for k,v in default_args.iteritems():
        
        if k is not "variables":
            if type(v) is list:
                parser.add_argument('--' + k, type=type(v[0]),nargs='+', default=v, help=k)
            elif type(v) is bool:
                parser.add_argument('--' + k, action='store_true', help=k)
            else:   
                parser.add_argument('--' + k, type=type(v), default=v, help=k)

    args = parser.parse_args()
    return args.__dict__

