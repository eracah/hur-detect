


import numpy as np
import sys



def convert_xy_offset_wh_scaled_box_to_xy_center_wh_box(xy_offset_wh_scaled_box,
                                                        xind, yind, scale_factor=32):
    xoff,yoff,w_scale, h_scale = xy_offset_wh_scaled_box
    x_center,y_center = scale_factor*(xind + xoff), scale_factor *(yind + yoff)

    w,h = 2.**w_scale * scale_factor, 2.**h_scale * scale_factor

    return [x_center,y_center,w,h]



def convert_xy_min_max_box_to_xy_offset_wh_scaled_box_with_inds(xy_min_max_box, scale_factor=32):
    xmin, xmax, ymin, ymax = xy_min_max_box[:4]
    x_center, y_center, w, h = _convert_xy_min_max_to_xy_center_wh(xmin, xmax, ymin, ymax)
    corner_inds, offset_scaled_box = convert_xy_center_wh_box_to_xy_offset_wh_scaled_box_with_inds(xy_center_wh_box, scale_factor=scale_factor )
    return corner_inds, offset_scaled_box
    
    
    
def convert_xy_center_wh_box_to_xy_offset_wh_scaled_box_with_inds(xy_center_wh_box, scale_factor=32 ):
    x_center, y_center, w, h = xy_center_wh_box[:4]
    
    x_corner_ind, y_corner_ind, x_offset, y_offset = get_xy_corner_inds_and_offsets(x_center, y_center, scale_factor)
    
    w_scaled_logged, h_scaled_logged = get_scaled_logged_wh(w, h, scale_factor)
    
    corner_inds = [x_corner_ind, y_corner_ind]
    offset_scaled_box = [x_offset, y_offset,w_scaled_logged, h_scaled_logged ]
    
    return corner_inds, offset_scaled_box
    
    

def get_xy_corner_inds_and_offsets(x,y, scale_factor):


        x_center_scaled, y_center_scaled = x_center / float(scale_factor), y_center / float(scale_factor)

        #take the floor of x and y -> which is rounding to nearest bottom left corner
        x_nearest_bottom_left_corner, y_nearest_bottom_left_corner = np.floor(x_center_scaled).astype("int"), np.floor(y_center_scaled).astype("int")


        x_offset, y_offset = x_center_scaled - x_nearest_bottom_left_corner, y_center_scaled - y_nearest_bottom_left_corner
        x_corner_ind, y_corner_ind = x_nearest_bottom_left_corner, y_nearest_bottom_left_corner

        return x_corner_ind, y_corner_ind, x_offset, y_offset


def get_scaled_logged_wh(w,h,scale_factor):
    
    
    w_scaled, h_scaled = w / float(scale_factor), h / float(scale_factor)
    w_scaled_logged, h_scaled_logged = np.log2(w_scaled), np.log2(h_scaled)
    
    return w_scaled_logged, h_scaled_logged
    





