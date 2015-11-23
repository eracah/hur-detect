__author__ = 'racah'
import h5py
import numpy as np
from operator import mul
import os
import glob
import sys

# 0 1 is hurriane
# 1 0  is not


# size of sampling window
w_size = 31
rad = w_size / 2

nclass = 2  #hur and nhur


#TODO HDF%ify this (get everything out of memory)
def run_tests(data_dict, hurs, pixels_to_use_per_image,cropped_ims, train_test_val_dist):

     ##assert len(tr_i) + len(te_i) + len(val_i) == num_images
    for name, X, i in [('tr', data_dict['X_train'], data_dict['tr_i']), ('val', data_dict['X_val'], data_dict['val_i']), ('te', data_dict['X_test'], data_dict['te_i'])]:
        assert X.shape[1:] == (hurs.shape[1] * w_size * w_size,), "%s not right shape: %s " % (name, str(X.shape[1:]))
        assert X.shape[0] == len(i) * pixels_to_use_per_image, "%s not right shape: %s " % (name, str(X.shape[0]))
    for name, i, y_arr in [('tr', data_dict['tr_i'], data_dict['y_train']), ('te', data_dict['te_i'], data_dict['y_test']), ('val', data_dict['val_i'], data_dict['y_val'])]:
        assert y_arr[y_arr[:, 0] == 1.].shape[0] > 0, "%s_%s_%s" % (name, y_arr.shape, len(i))
        assert y_arr[y_arr[:, 0] == 0.].shape[0] > 0

    assert cropped_ims.shape[0] == train_test_val_dist['num_images']
    print "Data Loader passed all tests!"

def set_up_train_test_val(hurs, boxes, im_indices, train_test_val_dist):
    tr_i, te_i, val_i = get_indices(im_indices, train_test_val_dist)

    X_train, y_train = generate_input_and_labels(hurs, tr_i, boxes)
    X_train, tr_means, tr_stds = normalize_each_channel(X_train)
    X_train = X_train.reshape(X_train.shape[0], reduce(mul, X_train.shape[1:]))

    X_test, y_test = generate_input_and_labels(hurs, te_i, boxes)
    X_test,_,_ = normalize_each_channel(X_test, tr_means, tr_stds)
    X_test = X_test.reshape(X_test.shape[0], reduce(mul, X_test.shape[1:]))

    X_val, y_val = generate_input_and_labels(hurs, val_i, boxes)
    X_val,_,_ = normalize_each_channel(X_val, tr_means, tr_stds)
    X_val = X_val.reshape(X_val.shape[0], reduce(mul, X_val.shape[1:]))
    return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test,'X_val':X_val, 'y_val':y_val}


def get_indices(im_indices, train_test_val_dist):
    d = train_test_val_dist
    te_i = im_indices[:d['test']]
    tr_i = im_indices[d['test']:d['test'] + d['train']]
    val_i = im_indices[d['test'] + d['train']:]
    return tr_i, te_i, val_i


def normalize_each_channel(arr, means=[], stds=[]):
    # assumes channels are on the axis 1
    if len(means) == 0:
        means = np.mean(arr, axis=(0, 2, 3))
    if len(stds) == 0:
        stds = np.std(arr, axis=(0, 2, 3))
    for channel, (mean, std) in enumerate(zip(means, stds)):
        arr[:, channel, :, :] -= mean
        arr[:, channel, :, :] /= std
    return arr, means, stds


def is_hurricane(x_cen, y_cen, xmin, ymin, xmax, ymax):
    xmin, ymin, xmax, ymax = [int(singleton) for singleton in [xmin, ymin, xmax, ymax]]
    assert isinstance(ymax, object)
    if x_cen >= xmin and x_cen <= xmax and y_cen >= ymin and y_cen <= ymax:
        return True
    else:
        return False


def get_pixel_index(x, y, cols):
    #based on the x,y coordinate, we get the pixel index
    return y * x - (cols - y)


def get_x_y_from_pixel(pixel_i, cols):
    rows_done = pixel_i / cols
    cols_done_in_unfinished_row = pixel_i % cols
    return cols_done_in_unfinished_row, rows_done

def save_preproc_data(preproc_data_dir, preproc_data_filename, data_dict):


    h5=h5py.File(os.path.join(preproc_data_dir,preproc_data_filename ), 'w')
    for k,v in data_dict.iteritems():
        h5.create_dataset(k, data=v)


    h5.close()



def generate_input_and_labels(im_array, im_indices, label_array):
    hurs = im_array
    boxes = label_array

    rows = im_array.shape[2]
    cols = im_array.shape[3]

    #subtract a frame of rad * rad around edge of image
    pixels_to_use_per_image = (cols - 2 * rad) * (rows - 2 * rad)
    total_pixels_to_use = len(im_indices) * pixels_to_use_per_image

    #number of totoal pixels to use by number of channels by window size
    inputs = np.zeros((total_pixels_to_use, hurs.shape[1], w_size, w_size))
    labels = np.zeros((total_pixels_to_use, 1))
    pixel_count = 0
    for im_index in im_indices:
        image = np.asarray(hurs[im_index])



        #mirror image so that
        #for each pixel
        for x in range(rad, rows - rad):
            for y in range(rad, cols - rad):
                x_min, x_max, y_min, y_max = (x - rad, x + rad, y - rad, y + rad )

                #print x_min, x_max,y_min,y_max
                #if window centered on this pixel stays in the frame
                assert (x_min >= 0 and x_max + 1 <= rows and y_min >= 0 and y_max + 1 <= cols), \
                    "Something is wrong with the pixels selected: %i,%i,%i,%i for %i window size" % (
                        x_min, x_max, y_min, y_max, w_size)

                #add window to inputs (need to reshape it to 1,channels,w_size,w_wize so it can be vstacked)
                cur_w = np.copy(image[:, x_min:x_max + 1, y_min:y_max + 1])
                #print pixel_count
                inputs[pixel_count] = cur_w  #.flatten()

                #label according to center pixel's label, so 0th element is 1 if hurricane for one-ht encoding and 1st elemnt if not hurricane
                #python unpacks along rows, so we transpose the row vector boxes[i], so it can be unpacked
                labels[pixel_count] = (1 if is_hurricane(x, y, *boxes[im_index].T) else 0)
                pixel_count += 1

    return inputs, labels

def get_newest_file_with_sufficient(train_test_val_dist, files_list, prefix):
    num_list = ['train', 'test', 'val']
    for file in files_list:
        nums = os.path.basename(file).split(prefix)[1].split('-')[1:4]
        nums = map(int, nums)

        #checks if num of images for train, test and val in file are greater than or equal to number requested
        hasEnoughData = all([train_test_val_dist[num_list[i]] >= num for i,num in enumerate(nums)])
        if hasEnoughData:
            return file
    return False

def load_hurricane(path, num_train=6,num_test_val=2, load_from_disk=True, preproc_data_dir='/global/project/projectdirs/nervana/evan/preproc_data'):
    print "starting..."
    do_compute=False
    train_test_val_dist = {'train': num_train, 'test': num_test_val, 'val': num_test_val, 'num_images': num_train + 2*num_test_val}
    if load_from_disk and os.path.exists(preproc_data_dir):
        path_basename = os.path.splitext(os.path.basename(path))[0]

        #get newest file that comes from same path and has same number of train specified
        filepaths= glob.iglob(os.path.join(preproc_data_dir, path_basename +'*'))
        filepath = get_newest_file_with_sufficient(train_test_val_dist, filepaths,path_basename)
        if not filepath:
            do_compute = True

        else:
            h5data = h5py.File(filepath)
            print "loading preproc data from file..."
            data_dict = {k : np.asarray(h5data[k]) for k in h5data.keys()}
            print 'done!'
    else:
        do_compute=True

    if do_compute:
        print 'computing preproc data'
        h5f = h5py.File(path)
        hurs = h5f['hurricane']
        boxes = h5f['hurricane_box']

        rows = hurs[0].shape[1]
        cols = hurs[0].shape[2]

        pixels_to_use_per_image = (cols - 2 * rad) * (rows - 2 * rad)

        im_indices = range(train_test_val_dist['num_images'])


        data_dict = set_up_train_test_val(hurs, boxes, im_indices, train_test_val_dist)

        #get the hurricane images we are using and crop them down to only the pixels that are used can be centered on the window
        #ie the pixels to use
        cropped_ims = hurs[im_indices, :, rad:rows - rad, rad:cols - rad]
        data_dict['cropped_ims'] = cropped_ims

        #run tests
        ind_tuple = get_indices(im_indices,train_test_val_dist)
        data_dict.update(zip(['tr_i','te_i','val_i'],ind_tuple))
        data_dict['boxes'] = boxes
        data_dict['nclass'] = nclass
        data_dict['w_size'] = w_size


        run_tests(data_dict, hurs, pixels_to_use_per_image, cropped_ims, train_test_val_dist)

        preproc_data_filename = '{0}-{1}-{2}-{3}-{4}'.format(os.path.splitext(os.path.basename(path))[0],
                                                                 data_dict['X_train'].shape[0] / pixels_to_use_per_image,
                                                                 data_dict['X_test'].shape[0] / pixels_to_use_per_image,
                                                                 data_dict['X_val'].shape[0] / pixels_to_use_per_image,
                                                                 w_size) +'.h5'
        if not os.path.exists(preproc_data_dir):
            os.mkdir(preproc_data_dir)
        save_preproc_data(preproc_data_dir,preproc_data_filename,data_dict)


    key_order = ['X_train', 'y_train', 'tr_i', 'X_test', 'y_test', 'te_i', 'X_val', 'y_val', 'val_i', 'nclass', 'w_size', 'cropped_ims', 'boxes']
    return [data_dict[k] for k in key_order]


if __name__ == "__main__":
    path = sys.argv[1]
    preproc_data_dir='./preproc_data'
    load_hurricane(path, 1, 1, True, preproc_data_dir)








