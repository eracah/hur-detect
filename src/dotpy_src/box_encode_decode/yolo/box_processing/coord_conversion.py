


import sys
import numpy as np



def convert_xy_center_wh_boxes_to_xy_min_max_boxes(xy_center_wh_boxes, xmax_val=768, ymax_val=768):
    '''takes list of boxes (each box is a list with x,y,w,h as first 4 and anything you want as rest of list) in xywh format
    and returns list in xmin,xmax,ymin,ymax format'''
    if len(xy_center_wh_boxes) == 0:
        return xy_center_wh_boxes
    xy_min_max_boxes = []
    assert_types(xy_center_wh_boxes) 
    
    for box in xy_center_wh_boxes:
        x_center,y_center,w,h = box[:4]
        rest = box[4:]
        xmin, xmax, ymin, ymax = _convert_xy_center_wh_box_to_xy_min_max_box(x_center,y_center,w,h, xmax_val, ymax_val)

        #keeps rest of item
        new_box = [xmin,xmax,ymin,ymax] + rest
        
        xy_min_max_boxes.append(new_box)
        
    return xy_min_max_boxes

def _convert_xy_center_wh_box_to_xy_min_max_box(x_center,y_center,w, h):
    xmin = x_center - w / 2.
    xmax = x_center + w / 2.
    ymin = y_center - h / 2.
    ymax = y_center + h / 2.
    return xmin, xmax, ymin, ymax
        

def convert_min_max_to_wh_boxes(xy_min_max_boxes):
    '''takes list of boxes in xmin,xmax,ymin,ymax format
    and returns list in xywh  format'''
    
    if len(xy_min_max_boxes) == 0:
        return xy_min_max_boxes
    xy_center_wh_boxes = []
    assert_types(xy_min_max_boxes)    
    for box in xy_min_max_boxes:
        xmin, xmax, ymin, ymax = box[:4]
        rest = box[4:]
        x_center, y_center, w, h = _convert_xy_min_max_to_xy_center_wh(xmin, xmax, ymin, ymax)

        new_box = [x_center, y_center, w, h]
        new_box.extend(rest)

        xy_center_wh_boxes.append(new_box)
        
    return xy_center_wh_boxes
    
def _convert_xy_min_max_to_xy_center_wh(xmin, xmax, ymin, ymax):
    '''takes list of boxes in xmin,xmax,ymin,ymax format
    and returns list in xywh  format'''
    w = xmax - xmin
    h = ymax - ymin
    x_center = xmin + w / 2.
    y_center = ymin + h / 2.
    return x_center, y_center, w, h




def assert_types(boxes):
    assert type(boxes) in [list, np.ndarray] and type(boxes[0]) in [list,np.ndarray], "can't help you here, input must be list of nonempty lists"
    
    



if __name__ == "__main__":

    xywh = [[50,50, 32,32], [100,200, 50, 70],[2.5,4.5,5,5],[4.5,2.5,3,3]]
    xyminmax = [[34,66, 34,66],[75,125,165,235],[0, 5.0, 2.0, 7.0],[3.0, 6.0, 1.0, 4.0]]
    

    xymm_guess = convert_xy_center_wh_boxes_to_xy_min_max_boxes(xywh)
    assert np.allclose(xymm_guess,xyminmax)
    print xymm_guess,xyminmax
    
    xywh_guess = convert_xy_min_max_boxes_to_xy_center_wh_boxes(xyminmax)
    assert np.allclose(xywh_guess, xywh)
    print xywh_guess, xywh
    
#     convert_xy_center_wh_boxes_to_xy_min_max_boxes([])
#     convert_xy_min_max_boxes_to_xy_center_wh_boxes([])
#     convert_xy_center_wh_boxes_to_xy_min_max_boxes(6)
#     convert_xy_min_max_boxes_to_xy_center_wh_boxes(4)
    



np.ndarray





