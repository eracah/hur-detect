


import tensorflow as tf
import sys
import numpy as np



def filter_out_negative_one_rows(array, axis=1):
    '''takes n x 5 tensor'''
    return array[np.any(array > -1,axis=axis)]



def extract_labels_from_box_array(box_array):
    pass



def make_box_coords_relative(bboxes, im_shape):
    box_list =[]
    for ind, bboxes_ex in enumerate(bboxes):
        bb = _make_box_coords_relative(bboxes_ex, im_shape)
        bb=np.expand_dims(bb, axis=0)
        box_list.append(bb)
    return np.concatenate(tuple(box_list))
        
def _make_box_coords_relative(bboxes_one_example, im_shape):
    """bboxes is [max_boxes x 5] where 5 is [ymin,xmin, ymax, xmax]
    y in this case is first index into matrix, which is y on a cartesian plane"""
    """we assume the dims for axis=1 are as follows:
          0: ymin
          1: xmin
          2: ymax
          3: xmax"""
    bboxes = bboxes_one_example.astype("float32")
    ymin_ind,ymax_ind, xmin_ind,  xmax_ind = range(4)
    ydim, xdim = im_shape
    num_boxes = bboxes[bboxes[:,0] > -1].shape[0]
    bboxes[:num_boxes, ymin_ind] /= float(ydim)
    bboxes[:num_boxes, ymax_ind] /= float(ydim)
    bboxes[:num_boxes, xmin_ind] /= float(xdim)
    bboxes[:num_boxes, xmax_ind] /= float(xdim)
    
    return bboxes



if __name__ == "__main__":
    sys.path.append("../../")
    from dotpy_src.load_data.get_generator import get_generator
    gen = get_generator("tr", batch_size=4)

    im, box = gen.next()

    print box

    #make_box_coords_relative(box,(768,1152))





