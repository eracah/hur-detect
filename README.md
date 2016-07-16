Instructions for running on Cori batch system:

* sbatch hur_main.sl \<epochs\> \<learning_rate\>

Instructions for running on a login node:

* module load python
* source activate deeplearning
* python hur_main.py -e \<epochs\> -l \<learning_rate\>

running a notebook:
* open up hur_main.ipynb
* run it

Prerequisites:
* lasagne
* theano
* scikit-learn
* h5py


