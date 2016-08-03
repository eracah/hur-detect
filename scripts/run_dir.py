
import matplotlib; matplotlib.use("agg")


import os



import sys



from shutil import copyfile
import imp



#print os.getcwd()



def create_run_dir(custom_rc=False):
    results_dir = os.getcwd() + '/results'
    run_num_file = os.path.join(results_dir, "run_num.txt")
    if not os.path.exists(results_dir):
        print "making results dir"
        os.mkdir(results_dir)

    if not os.path.exists(run_num_file):
        print "making run num file...."
        f = open(run_num_file,'w')
        f.write('0')
        f.close()




    f = open(run_num_file,'r+')

    run_num = int(f.readline()) + 1

    f.seek(0)

    f.write(str(run_num))


    run_dir = os.path.join(results_dir,'run%i'%(run_num))
    os.mkdir(run_dir)
    
    if custom_rc:
        make_custom_config_file(run_dir)
    return run_dir



def make_custom_config_file(run_dir):

    theano_path = imp.find_module('theano')[1]

    copyfile(os.path.join(theano_path,'.theanorc'), os.path.join(run_dir, '.theanorc'))

    #make the compile dir be this unique run dir to lock issues
    with open(os.path.join(run_dir, '.theanorc'), "a") as f:
        f.write('[base_compiledir]')
        f.write(run_dir)
      
    #point to new config file
    os.environ['THEANORC'] = os.path.join(run_dir, '.theanorc')
    f.close()





