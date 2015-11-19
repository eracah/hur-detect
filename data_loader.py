__author__ = 'racah'
import h5py
import numpy as np
from operator import mul
import os
# 0 1 is hurriane
# 1 0  is not

# size of sampling window
w_size = 31
rad = w_size / 2
train_test_val_dist = {'train': 6, 'test': 2, 'val': 2, 'num_images': 10}
nclass = 2  #hur and nhur
preproc_data_dir='./preproc_data'
if not os.path.exists(preproc_data_dir):
    os.mkdir(preproc_data_dir)

def set_up_train_test_val(hurs, boxes, im_indices):
    tr_i, te_i, val_i = get_indices(im_indices)

    X_train, y_train = generate_input_labels(hurs, tr_i, boxes)
    X_train, tr_means, tr_stds = normalize_each_channel(X_train)
    X_train = X_train.reshape(X_train.shape[0], reduce(mul, X_train.shape[1:]))

    X_test, y_test = generate_input_labels(hurs, te_i, boxes)
    X_test,_,_ = normalize_each_channel(X_test, tr_means, tr_stds)
    X_test = X_test.reshape(X_test.shape[0], reduce(mul, X_test.shape[1:]))

    X_val, y_val = generate_input_labels(hurs, val_i, boxes)
    X_val,_,_ = normalize_each_channel(X_val, tr_means, tr_stds)
    X_val = X_val.reshape(X_val.shape[0], reduce(mul, X_val.shape[1:]))
    return X_train, y_train, X_test, y_test, X_val, y_val


def get_indices(im_indices):
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

def save_preproc_data(preproc_data_filename, data_list):
    (X_train, y_train, tr_i), \
    (X_test, y_test, te_i), \
    (X_val, y_val, val_i), \
    nclass, \
    w_size, \
    cropped_ims, \
    boxes = data_list

    h5=h5py.File(os.path.join(preproc_data_dir,preproc_data_filename ), 'w')
    h5.create_dataset('X_train', data=X_train)
    h5.create_dataset('X_test', data=X_test)
    h5.create_dataset('X_val', data=X_val)
    h5.create_dataset('tr_i', data=tr_i)
    h5.create_dataset('te_i', data=te_i)
    h5.create_dataset('val_i', data=val_i)
    h5.create_dataset('w_size', data=w_size)
    h5.create_dataset('cropped_ims', data=cropped_ims)
    h5.create_dataset('boxes', data=boxes)
    h5.close()



def generate_input_labels(im_array, im_indices, label_array):
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


def load_hurricane(path):
    h5f = h5py.File(path)
    hurs = h5f['hurricane']
    boxes = h5f['hurricane_box']

    rows = hurs[0].shape[1]
    cols = hurs[0].shape[2]

    pixels_to_use_per_image = (cols - 2 * rad) * (rows - 2 * rad)
    im_indices = range(train_test_val_dist['num_images'])

    X_train, y_train, X_test, y_test, X_val, y_val = set_up_train_test_val(hurs, boxes, im_indices)

    #get the hurricane images we are using and crop them down to only the pixels that are used can be centered on the window
    #ie the pixels to use
    cropped_ims = hurs[im_indices, :, rad:rows - rad, rad:cols - rad]


    #run tests
    tr_i, te_i, val_i = get_indices(im_indices)
    ##assert len(tr_i) + len(te_i) + len(val_i) == num_images
    for name, X, i in [('tr', X_train, tr_i), ('val', X_val, val_i), ('te', X_test, te_i)]:
        assert X.shape[1:] == (hurs.shape[1] * w_size * w_size,), "%s not right shape: %s " % (name, str(X.shape[1:]))
        assert X.shape[0] == len(i) * pixels_to_use_per_image, "%s not right shape: %s " % (name, str(X.shape[0]))
    for name, i, y_arr in [('tr', tr_i, y_train), ('te', te_i, y_test), ('val', val_i, y_val)]:
        assert y_arr[y_arr[:, 0] == 1.].shape[0] > 0, "%s_%s_%s" % (name, y_arr.shape, len(i))
        assert y_arr[y_arr[:, 0] == 0.].shape[0] > 0

    assert cropped_ims.shape[0] == train_test_val_dist['num_images']
    print "Data Loader passed all tests!"

    preproc_data_filename = '{0}-{1}-{2}-{3}-{4}-{5}'.format(os.path.splitext(os.path.basename(path))[0], X_train.shape[0] / pixels_to_use_per_image, X_train.shape[0], X_test.shape[0], X_val.shape[0], w_size) +'.h5'
    data_list = [(X_train, y_train, tr_i), (X_test, y_test, te_i), (X_val, y_val, val_i), nclass, w_size, cropped_ims, boxes]
    save_preproc_data(preproc_data_filename,data_list)


    return (X_train, y_train, tr_i), (X_test, y_test, te_i), (X_val, y_val, val_i), nclass, w_size, cropped_ims, boxes








