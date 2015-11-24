__author__ = 'racah'
from neon.util.argparser import NeonArgparser
import os
#TODO log on to edison and run
# 01 is hur 10 is nhur
# parse the command line arguments
parser = NeonArgparser(__doc__)
final_dir = './results'
model_files_dir = './model_files'
images_dir = './images'
dirs = [model_files_dir, final_dir, images_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)

parser.add_argument('--load_data_from_disk')
parser.add_argument('--h5file')
parser.add_argument('--num_train')
parser.add_argument('--num_test_val')
parser.add_argument('--preproc_data_dir')

parser.set_defaults(batch_size=1000,
                    test=False,
                    #save_path=model_files_dir,
                    h5file='/global/project/projectdirs/nervana/yunjie/dataset/localization_test/expand_hurricanes_loc.h5',
                    serialize=2,
                    epochs=100,
                    progress_bar=True,
                    datatype='f64',
                    model_file=False,
                    just_test=False,
                    eval_freq=1,
                    load_data_from_disk=1,
                    num_train=6,
                    num_test_val=2,
                    preproc_data_dir='/global/project/projectdirs/nervana/evan/preproc_data')

args = parser.parse_args()
args.load_data_from_disk = bool(int(args.load_data_from_disk))