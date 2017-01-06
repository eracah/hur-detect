#Instructions for running on Cori batch system:

* sbatch hur_main.sl #and the below command line arguments

#Instructions for running from terminal:

* module load deeplearning #or if you are not on cori, put numpy, lasagne, theano, scikit-learn, h5py on your path
* python hur_main.py  #and the below command line args


#Command line args

  --shuffle SHUFFLE     shuffle (default: False)
  
  --dropout_p DROPOUT_P
                        dropout_p (default: 0)
                        
  --get_ims GET_IMS     get_ims (default: False)
  
  --labels_only LABELS_ONLY
                        labels_only (default: True)
                        
  --yolo_load_path YOLO_LOAD_PATH
                        yolo_load_path (default: None)
                        

                        
  --num_extra_conv NUM_EXTRA_CONV
                        num_extra_conv (default: 0)
                        
  --save_path SAVE_PATH
                        save_path (default: None)
                        
  --filters_scale FILTERS_SCALE
                        filters_scale (default: 1.0)
                        
  
  --scale_factor SCALE_FACTOR
                        scale_factor (default: 64)
                        
  --num_ims_to_plot NUM_IMS_TO_PLOT
                        num_ims_to_plot (default: 8)
                        
  --input_shape INPUT_SHAPE
                        input_shape (default: (None, 16, 768, 1152))
                        
  --save_weights SAVE_WEIGHTS
                        save_weights (default: True)
                        
  --use_fc USE_FC       use_fc (default: False)
  
  --filter_dim FILTER_DIM
                        filter_dim (default: 5)
                        
  --coord_penalty COORD_PENALTY
                        coord_penalty (default: 5)
                        
  --box_sizes BOX_SIZES
                        box_sizes (default: [(64, 64)])
                        
  --val_years VAL_YEARS
                        val_years (default: [1982, 1986])
                        
  --metadata_dir METADATA_DIR
                        metadata_dir (default:
                        /storeSSD/eracah/data/metadata/)
                        
  --batch_norm BATCH_NORM
                        batch_norm (default: False)
                        
  --tr_years TR_YEARS   tr_years (default: [1979, 1980, 1981, 1983, 1985,
                        1987])
                        
  --epochs EPOCHS       epochs (default: 10000)
  
  --size_penalty SIZE_PENALTY
                        size_penalty (default: 7)
                        
  --num_layers NUM_LAYERS
                        num_layers (default: 6)
                        
  --num_test_days NUM_TEST_DAYS
                        num_test_days (default: 365)
                        
  --weight_decay WEIGHT_DECAY
                        weight_decay (default: 0.0005)
                        
  --3D 3D               3D (default: False)
  
  --ae_load_path AE_LOAD_PATH
                        ae_load_path (default: None)
                        
  --lambda_ae LAMBDA_AE
                        lambda_ae (default: 10)
                        
  --data_dir DATA_DIR   data_dir (default: /storeSSD/eracah/data/netcdf_ims)
  
  --test TEST           test (default: False)

  --time_chunks_per_example TIME_CHUNKS_PER_EXAMPLE
                        time_chunks_per_example (default: 1)
                        
                        
  --no_plots NO_PLOTS   no_plots (default: False)
  
  --learning_rate LEARNING_RATE
                        learning_rate (default: 0.0001)
                        
  --batch_size BATCH_SIZE
                        batch_size (default: 1)
                        
  --get_fmaps GET_FMAPS
                        get_fmaps (default: False)
                        
  --yolo_batch_norm YOLO_BATCH_NORM
                        yolo_batch_norm (default: True)
                        
  --test_years TEST_YEARS
                        test_years (default: [1984])
                        
  --iou_thresh IOU_THRESH
                        iou_thresh (default: 0.1)
                        
  --num_classes NUM_CLASSES
                        num_classes (default: 4)

  --conf_thresh CONF_THRESH
                        conf_thresh (default: 0.8)
                        
  --nonobj_penalty NONOBJ_PENALTY
                        nonobj_penalty (default: 0.5)
                        
  --num_tr_days NUM_TR_DAYS
                        num_tr_days (default: 365)
                        
  --ignore_plot_fails IGNORE_PLOT_FAILS
                        ignore_plot_fails (default: 1)

#running the notebook:
* open up hur_main.ipynb
* run it

#Prerequisites:
* lasagne
* theano
* scikit-learn
* h5py


