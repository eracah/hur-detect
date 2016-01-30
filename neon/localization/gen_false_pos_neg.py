# plotting false positives script
from data_models.model import AR
from data_models.hurricane import Hurricane
import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib import cm
import os
from skimage.filters import roberts, sobel, scharr
# from scipy import ndimage.filters.sobel

import copy


def get_file_metadata(te_inf_file):
    weath_type, _, _ = te_inf_file.split('-')
    return weath_type


def get_raw_images(rec_file):
    rec_f = pickle.load(open(rec_file))
    if isinstance(rec_f, dict):
        raw_images = get_raw_images_rec(rec_f)

    elif isinstance(rec_f, np.ndarray):
        return rec_f.squeeze()

    else:
        raw_images = get_raw_images_exp_obj(rec_f)
    return raw_images


def get_raw_images_rec(rec_dict):
    r = rec_dict
    raw_train, raw_test = Hurricane(training_size=r['tr_size'],
                                        test_size=r['te_size'],
                                        repo_path='./results',
                                        data_path='./data',
                                        hdf5_file=r['h5_file']
        ).get_raw_train_test(r['seed'])

    # elif rec_dict['type'] == 'ar':
    #     raw_train, raw_test, _, _ = AR(fland=r['fland']
    #     far=r['far'], repo_path=repo_path) .get_raw_train_test(seed=r['seed'])
    return raw_test


def get_raw_images_exp_obj(rec_f):
    #get dataset from joaquins
    pass


def get_tp_fp_fn_indices(te_inf_file, te_tgt_file):
    predictions = pickle.load(open(te_inf_file))
    truth = pickle.load(open(te_tgt_file))


    #rounded predictions to one hot incoding, so [0.51, 0.49] becomes [1,0], etc.
    predictions_r = np.asarray([[1., 0.] if row[0] > row[1] else [0., 1.] for row in predictions])

    diff = predictions_r - truth

    d = np.arange(diff.shape[0])

    #prediction is [0,1] and truth is [1,0] -> false negatuve
    fn_i = d[diff[:, 1] == 1]  # and diff[:,0] == -1

    #prediction is [1,0] and truth is [0,1] -> false positive
    fp_i = d[diff[:, 0] == 1]  # and diff[:,1] == -1

    pred_pos = np.logical_and(predictions_r[:, 1] == 0, predictions_r[:, 0] == 1)
    tgt_pos = np.logical_and(truth[:, 1] == 0, truth[:, 0] == 1)
    tp_i = np.logical_and(pred_pos, tgt_pos)

    return {'tp': tp_i, 'fp': fp_i, 'fn': fn_i}


def save_jpgs_hr(raw_images, image_ind_d, des_keys, output_path, max_images=10):
    base_save_path = os.path.join(output_path, 'images')
    keys = ['TMQ', 'V850', 'PSL', 'U850', 'T500', 'UBOT', 'T200', 'VBOT' ]
    des_idxs = [idx for idx, key in enumerate(keys) if key in des_keys]
    for im_type in image_ind_d.keys():
        save_path = base_save_path
        dirs = ['', 'HR', im_type]
        for dir in dirs:
            save_path = os.path.join(save_path, dir)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)

        images = raw_images[image_ind_d[im_type]]
        print images.shape[0]
        if images.shape[0] > max_images:
            images = images[:max_images]
        print images.shape[0]

        for idx, image in enumerate(images):
            x,y =np.meshgrid(np.arange(image.shape[1]),np.arange(image.shape[2]))


            fig, ax = plt.subplots()
            aa = ax.imshow(image[keys.index('PSL')], cmap=cm.jet)
            plt.colorbar(aa, ax=ax, orientation='horizontal', shrink=0.7)
            ax.streamplot(x, y, image[keys.index('UBOT')],image[keys.index('VBOT')],density=3)
            im_name =  'HR_' + im_type + '_' + str(idx) + '_'  + '.png'

            plt.savefig(os.path.join(save_path, im_name), dpi=32)
            plt.close(fig)



def save_jpgs_ar(raw_images, image_ind_d, landmask_path, output_path, country, max_images=10):
    base_save_path = os.path.join(output_path, 'images')
    landmask = np.flipud(np.asarray(pickle.load(open(landmask_path, 'r'))['mask']).squeeze())
    lm_im = roberts(landmask)
    lm_im[lm_im > 0.1] = 1

    lm_im = np.ma.masked_where(lm_im == 0, lm_im)
    lm_im[lm_im > 0] = 1




    # my_cmap = copy.copy(plt.cm.get_cmap('gray'))
    # my_cmap.set_bad(alpha=0)
    #
    #
    # lm_im[lm_im < 0.1] = np.nan
    # lm_im[lm_im > 0.1] = 1.



    for im_type in image_ind_d.keys():
        save_path = base_save_path
        dirs = ['', 'AR_' + country, im_type]
        for dir in dirs:
            save_path = os.path.join(save_path, dir)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
        images = raw_images[image_ind_d[im_type]]
        print images.shape[0]
        if im_type == 'tp' and images.shape[0] > max_images:
            images = images[:max_images]
        print images.shape[0]

        for idx, image in enumerate(images):
            image = np.flipud(image)
            fig, ax = plt.subplots()
            aa = ax.imshow(image, cmap=cm.jet)
            ax.imshow(lm_im,cmap=cm.binary, interpolation='nearest')
            # ax.imshow(lm_im, cmap=my_cmap, alpha=0.7 )
            plt.colorbar(aa, ax=ax, orientation='horizontal')
            plt.savefig(os.path.join(save_path, 'AR_' + country + '_' + im_type + '_' + str(idx) + '.png'))
            plt.close(fig)


def gen_false_pos_neg_hr(te_inf_file, te_tgt_file, re_creation_file_path, output_path, des_keys):
    image_ind_d = get_tp_fp_fn_indices(te_inf_file, te_tgt_file)
    raw_images = get_raw_images(re_creation_file_path)
    save_jpgs_hr(raw_images, image_ind_d, des_keys, output_path)


def gen_false_pos_neg_ar(te_inf_file, te_tgt_file, landmask_path, re_creation_file_path, output_path, country):
    raw_images = get_raw_images(re_creation_file_path)
    image_ind_d = get_tp_fp_fn_indices(te_inf_file, te_tgt_file)
    save_jpgs_ar(raw_images, image_ind_d, landmask_path, output_path, country)

if __name__ == "__main__":
    # basepath = './results/HR/8_4_2015_16_18'
    # te_inf_file = os.path.join(basepath, 'Hurricane/test-inference.pkl')
    # te_tgt_file = os.path.join(basepath, 'Hurricane/test-targets.pkl')
    # re_creation_file_path = os.path.join(basepath, 're_creation.pkl')
    # output_path = './results'
    # des_keys = ['PSL', 'UBOT', 'VBOT']
    # gen_false_pos_neg_hr(te_inf_file, te_tgt_file, re_creation_file_path, output_path, des_keys)


    '''---------'''

    basepath = './results/EU/'
    te_inf_file = os.path.join(basepath, 'AGU_test-inference.pkl')
    te_tgt_file = os.path.join(basepath, 'AGU_test-targets.pkl')
    re_creation_file_path = os.path.join(basepath,'AGU_eu_test.pkl')
    landmask_path = './data/landmask_imgs_eu.pkl'
    output_path = './'
    country = 'eu'
    gen_false_pos_neg_ar(te_inf_file, te_tgt_file, landmask_path, re_creation_file_path, output_path, country)



    #
    # basepath = './results/US/'
    # te_inf_file = os.path.join(basepath, 'AGU_test-inference.pkl')
    # te_tgt_file = os.path.join(basepath, 'AGU_test-targets.pkl')
    # re_creation_file_path = os.path.join(basepath,'AGU_us_test.pkl')
    # landmask_path = './data/landmask_imgs_us.pkl'
    #
    # output_path = './'
    # country = 'us'
    # gen_false_pos_neg_ar(te_inf_file, te_tgt_file, landmask_path, re_creation_file_path, output_path, country)





