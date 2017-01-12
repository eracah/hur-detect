
import matplotlib; matplotlib.use("agg")


import sys
from helper_fxns import *
from run_dir import *
from lasagne.nonlinearities import *
from lasagne.init import *



default_args = {                  'learning_rate': 0.0001,
                                  'num_tr_days': 365,
                                  'input_shape': (None,16,768,1152),
                                  'dropout_p': 0, 
                                  'weight_decay': 0.0005, 
                                  'num_layers': 6,
                                  'num_extra_conv': 0,
                                  'momentum': 0.9,
                                  'lambda_ae' : 10,
                                  'coord_penalty': 5,
                                  'size_penalty': 7,
                                  'nonobj_penalty': 0.5,
                                  'iou_thresh' : 0.5,
                                  'conf_thresh': 0.8,
                                  'shuffle': False,
                                  "use_fc": False,
                                  'metadata_dir': "/home/evan/data/climate/labels/",
                                  'data_dir': "/home/evan/data/climate/input",
                                  'batch_size' : 1,
                                  'epochs': 10000,
                                  'tr_years': [1979],
                                  'val_years': [1980],
                                  "test_years" : [1984],
                                  'save_weights': True,
                                  'num_classes': 4,
                                  'labels_only': True,
                                  'time_chunks_per_example': 1,
                                  'filter_dim':5,
                                  'scale_factor': 64,
                                  'nonlinearity': LeakyRectify(0.1),
                                  'w_init': HeUniform(),
                                  "batch_norm" : False,
                                  "num_ims_to_plot" : 8,
                                  "test": False,
                                  "get_fmaps": False,
                                  "grid_search": False,
                                  "yolo_batch_norm" : True,
                                  "filters_scale" : 1.,
                                  "yolo_load_path": "None",
                                  "no_plots": False,
                                  "3D": False,
                                  "get_ims": False,
                                  "save_path":"None",
                                  'num_test_days':365,
                                  'box_sizes':[(64,64)],
                                  "ignore_plot_fails":1,
                                  "ae_load_path": "None",
                                  "variables": [u'PRECT',u'PS',u'PSL',
                                                 u'QREFHT',
                                                 u'T200',
                                                 u'T500',
                                                 u'TMQ',
                                                 u'TREFHT',
                                                 u'TS',
                                                 u'U850',
                                                 u'UBOT',
                                                 u'V850',
                                                 u'VBOT',
                                                 u'Z1000',
                                                 u'Z200',
                                                 u'ZBOT'],
                                "xdim": 768,
                                "ydim": 1152,
                                "time_steps_per_day": 8,
                                "time_steps_per_file": 8,
                                "max_files_open": 45,
                                "time_step_stride": 2,

                                  
                    }



def process_kwargs(args={},save_res=True):
    kwargs = default_args
    
    if "save_path" in args:
        if args.save_path == "None":
            save_path = None
        else:
            save_path = args.save_path
        kwargs.update(args)
    else:
        save_path=None


    if kwargs["lambda_ae"] == 0:
        kwargs["labels_only"] = True
    if kwargs["3D"] == True:
        kwargs["labels_only"] = False
        kwargs["input_shape"] = (None,16,8,768,1152)
        kwargs['time_chunks_per_example'] = 8

    kwargs['num_val_days'] = int(np.ceil(0.2*kwargs['num_tr_days']))

    
    if save_res:
        run_dir = create_run_dir(save_path)
        kwargs['save_path'] = run_dir
        '''save hyperparams'''
        dump_hyperparams(kwargs,run_dir)

        kwargs["logger"] = setup_logging(kwargs['save_path'])
    return kwargs





