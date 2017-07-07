# hur-detect
Deep Semi-Supervised Object Detection for Extreme Weather Events


## Getting Most Stable Version

Last fully functional commit: 553448aeeaf7bda4bec7d4ec78a9624991c1c4d

    git reset 553448aeeaf7bda4bec7d4ec78a9624991c1c4d
## Running Code

### Prereqs

* tensorflow >= 1.0
* keras 2.0
* scikit-learn
* numpy

### Running

    module load deeplearning # if on Cori

    python main.py < command line arguments to main.py>

### Monitoring with Tensorboard

    tensorboard --logdir ./logs


### Recommended Command Line Arguments to main.py
  --labels_file
  * **default**: /home/evan/data/climate/csv_labels/labels_no_negatives.csv
  
  * **recommended**: /project/projectdirs/dasrepo/www/climate/data/h5data/labels_no_negatives.csv
                        
  --logs_dir
  * **recommended**: ./logs

  --tr_data_file
                        
  * **default**: /home/evan/data/climate/climo_1980.h5
  * **recommended**: /project/projectdirs/dasrepo/www/climate/data/h5data/climo_1980.h5
                        
                        
   --val_data_file 
                        
   * **default**: /home/evan/data/climate/climo_1981.h5
   * **recommended**: /project/projectdirs/dasrepo/www/climate/data/h5data/climo_1981.h5
                        
   --test_data_file
                        
   * **default**: /home/evan/data/climate/climo_1982.h5
   * **recommended**: /project/projectdirs/dasrepo/www/climate/data/h5data/climo_1982.h5
    



### Optional Command Line Arguments to main.py
  -h, 
  --help            show the help message and exit
  
  --min_delta MIN_DELTA
                        min_delta (default: 0.001)
  
  --num_max_boxes NUM_MAX_BOXES
                        num_max_boxes (default: 15)
  
  --data_name DATA_NAME
                        data_name (default: climate)
  
  --anchor_sizes ANCHOR_SIZES [ANCHOR_SIZES ...]
                        anchor_sizes (default: [(20.48, 51.2), (51.2, 133.12),
                        (133.12, 215.04), (215.04, 296.96), (296.96, 378.88),
                        (378.88, 460.8), (460.8, 542.72)])
  
  --num_tr_ims NUM_TR_IMS
                        num_tr_ims (default: 8)
  
  --patience PATIENCE   patience (default: 5)
  
  --alpha ALPHA         alpha (default: 1.0)
  
  --anchor_size_bounds ANCHOR_SIZE_BOUNDS [ANCHOR_SIZE_BOUNDS ...]
                        anchor_size_bounds (default: [0.1, 0.9])
  
  --num_epochs NUM_EPOCHS
                        num_epochs (default: 10)
  
  --scale_factor SCALE_FACTOR
                        scale_factor (default: 32)
  
  --default_negatives DEFAULT_NEGATIVES
                        default_negatives (default: 5)
  
  --optimizer OPTIMIZER
                        optimizer (default: adam)
  
  --matching_threshold MATCHING_THRESHOLD
                        matching_threshold (default: 0.5)
  
  --ydim YDIM           ydim (default: 1152)
  
  --select_threshold SELECT_THRESHOLD
                        select_threshold (default: 0.01)
  
  --exp_name EXP_NAME   exp_name (default: None)
  
  --prior_scaling PRIOR_SCALING [PRIOR_SCALING ...]
                        prior_scaling (default: [0.1, 0.1, 0.2, 0.2])
  
  --lr LR               lr (default: 0.0001)
  
  --select_top_k SELECT_TOP_K
                        select_top_k (default: 400)
  
  --anchor_steps ANCHOR_STEPS [ANCHOR_STEPS ...]
                        anchor_steps (default: [8, 16, 32, 64, 128, 256, 512])
  
  --gpu GPU             gpu (default: 0)
  
  --w_init W_INIT       w_init (default: he_normal)
  
  --label_smoothing LABEL_SMOOTHING
                        label_smoothing (default: 0.0)
  
  --feat_shapes FEAT_SHAPES [FEAT_SHAPES ...]
                        feat_shapes (default: [(96, 144), (48, 72), (24, 36),
                        (12, 18), (6, 9), (3, 5), (1, 1)])
  
  --num_test_ims NUM_TEST_IMS
                        num_test_ims (default: 32)
  

  
  --anchor_offset ANCHOR_OFFSET
                        anchor_offset (default: 0.5)
  
  --normalizations NORMALIZATIONS [NORMALIZATIONS ...]
                        normalizations (default: [20, -1, -1, -1, -1, -1, -1])
  
  --anchor_box_size ANCHOR_BOX_SIZE
                        anchor_box_size (default: 32)
  
  --raw_input_shape RAW_INPUT_SHAPE
                        raw_input_shape (default: (16, 768, 1152))
  
  --keep_top_k KEEP_TOP_K
                        keep_top_k (default: 200)
  
  --input_shape INPUT_SHAPE
                        input_shape (default: (16, 768, 1152))
  
  --xdim XDIM           xdim (default: 768)
  

  
  --beta BETA           beta (default: 1.0)
  
  --batch_size BATCH_SIZE
                        batch_size (default: 4)
  
  --autoencoder_weight AUTOENCODER_WEIGHT
                        autoencoder_weight (default: 0.1)
  
  --fit_name FIT_NAME   fit_name (default: tf_fit)
  

  
  --nms_threshold NMS_THRESHOLD
                        nms_threshold (default: 0.45)
  
  --num_classes NUM_CLASSES
                        num_classes (default: 4)
  
  --w_decay W_DECAY     w_decay (default: 0.0005)
  
  --box_sizes BOX_SIZES [BOX_SIZES ...]
                        box_sizes (default: [(64, 64)])
  
  --detection_model DETECTION_MODEL
                        detection_model (default: ssd)
  

  
  --num_val_ims NUM_VAL_IMS
                        num_val_ims (default: 8)
  
  --base_model BASE_MODEL
                        base_model (default: vgg16)
  
  --momentum MOMENTUM   momentum (default: 0.9)
  
  --anchor_ratios ANCHOR_RATIOS [ANCHOR_RATIOS ...]
                        anchor_ratios (default: [[2, 0.5], [2, 0.5, 3,
                        0.3333333333333333], [2, 0.5, 3, 0.3333333333333333],
                        [2, 0.5, 3, 0.3333333333333333], [2, 0.5, 3,
                        0.3333333333333333], [2, 0.5], [2, 0.5]])
  
  --negative_ratio NEGATIVE_RATIO
                        negative_ratio (default: 3)
  

  
  --tf_format           tf_format (default: True)
