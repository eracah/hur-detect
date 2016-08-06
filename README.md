#Instructions for running on Cori batch system:

* sbatch hur_main.sl #and the below command line arguments

#Instructions for running from terminal:

* module load deeplearning #or if you are not on cori, put numpy, lasagne, theano, scikit-learn, h5py on your path
* python hur_main.py  #and the below command line args


#Command line args

-e EPOCHS, --epochs EPOCHS

  -l LEARN_RATE, --learn_rate LEARN_RATE
  
  -n NUM_IMS, --num_ims NUM_IMS
  
  -f NUM_FILTERS, --num_filters NUM_FILTERS
  
  --fc      number of fully connected units
  --coord_penalty COORD_PENALTY
                        penalty for guessing coordinates wrong
                        
  --size_penalty SIZE_PENALTY
                        penalty for guessing height or width wrong
                        
  --nonobj_penalty NONOBJ_PENALTY
                        penalty for guessing an object where one isnt

  -c NUM_EXTRA_CONV, --num_extra_conv NUM_EXTRA_CONV
                        conv layers to add on to each conv layer before max
                        pooling
                        
                        
  --num_convpool NUM_CONVPOOL
                        number of conv layer-pool layer pairs
                        
                        
                        
  --momentum MOMENTUM   momentum

#running the notebook:
* open up hur_main.ipynb
* run it

#Prerequisites:
* lasagne
* theano
* scikit-learn
* h5py


