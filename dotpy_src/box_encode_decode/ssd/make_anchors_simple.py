


import numpy as np



def create_default_box_shapes_for_feature_map(fmap_receptive_field_size, filter_size):
    rf = fmap_receptive_field_size
    """takes in
           int: receptive_field_size (1 for orginal image, 2 for downsampled by 2 image,... n for downsampled by n image )
           int: filter_size: size of filter (we assume filters are square) 
        returns list of 2d tuples which contain (width, height) of box"""
    """by default lets do a (rf = receptive field size):
            * squares: rf x rf box, 2rf x 2rf,..., (filter_size -1) * rf x (filter_size -1)*rf box
            * rectangle rf x 2rf, 2rf x rf,..., (filter_size -1) * rf x rf, rf x (filter_size -1) * rf
        so the larger the receptive field -> the larger the context for the box aka the larger the filter_size * rf to box size ratio
     """
    default_boxes = []
    
    sq_boxes = make_square_boxes(rf, filter_size)
    default_boxes.extend(sq_boxes)
    
    rect_boxes = make_rectangle_boxes(rf, filter_size)
    default_boxes.extend(rect_boxes)
    #get rid of duplicated
    default_boxes = list(set(default_boxes))
    return default_boxes
        
        
        

def make_square_boxes(receptive_field_size, filter_size):
    rf = receptive_field_size
    sq_boxes = []
    # every box from [rf,filter_size*rf) but only even boxes (so every 2)
    for dim in range(receptive_field_size, filter_size*receptive_field_size, receptive_field_size):
        sq_boxes.append((dim,dim))
    return sq_boxes

def make_rectangle_boxes(receptive_field_size, filter_size):
    rf = receptive_field_size
    rect_boxes = []
    for first_dim in range(receptive_field_size, filter_size*receptive_field_size, receptive_field_size):
        for second_dim in range(filter_size*receptive_field_size,receptive_field_size, -receptive_field_size):
            if first_dim != second_dim:
                rect_box = (first_dim, second_dim)
                if rect_box not in rect_boxes:
                    rect_boxes.append(rect_box)
    return rect_boxes



def make_empty_box_tensor(box_shapes, box_lists, xdim,ydim, num_channels_for_one_box):
    num_default_boxes_per_cell = len(box_shapes)
    num_examples = len(box_lists)
    num_channels_in_gt_box_tensor = num_channels_for_one_box * num_default_boxes_per_cell
    gt_box_tensor = np.zeros((num_examples, num_channels_in_gt_box_tensor, xdim, ydim))
    return gt_box_tensor
    

def does_box_get_cutoff(xmin,ymin,xmax,ymax, frame_xmax, frame_ymax):
    is_cutoff = False
    if xmin < 0 and ymin < 0:
        if xmax > frame_xmax and ymax > frame_ymax:
            is_cutoff=True
    return is_cutoff

def match_boxes(default_box, gt_boxes, thresh=0.5):
    #is gt_boxes xywh or xmin,xmax,...?
    xoff,yoff, scaled_w, scaled_h, cls = 5*[0.]
    ious = [get_iou(default_box, gt_box) for gt_box in gt_boxes]
    max_iou = np.max(ious)
    if max_iou >= thresh:
        max_iou_ind = np.argmax(ious)
        best_gt = gt_boxes[max_iou_ind]
        xoff, yoff = calc_xyoff(default_box, best_gt)
        scaled_w, scaled_h = calc_wh_scale(default_box, best_gt)
        is_match = True
        
    else:
        is_match = False
    return is_match, xoff,yoff, scaled_w, scaled_h, cls
        
    
    

def calc_xyoff(default_box, best_gt):
    dx, dy = default_box[:2]
    gx, gy = best_gt[:2]
    xoff = (gx - dx) / float(dx)
    yoff = (gy - dy) / float(dy)
    return xoff,yoff
    

def calc_wh_scale(default_box, best_gt):
    dw, dh = default_box[2:4]
    gw, gh = best_gt[2:4]
    scaled_w  = np.log2(float(gw) / dw)
    scaled_h  = np.log2(float(gh) / dh)
    return scaled_w, scaled_h
    

def create_default_boxes_for_feature_map(box_lists, feature_map_size, receptive_field_size, filter_size, use_cutoff_boxes=False):
    
    """ takes in:
                 feature map size (tuple): (xdim,ydim)
                 receptive field size (int)
                 filter_size (int): we assume square filters for now
                 use_cutoff_boxes (bool): whether to have boxes that get cut off from image"""
    boxes = []
    box_shapes = create_default_box_shapes_for_feature_map(receptive_field_size, filter_size)
    num_channels_for_one_box = 4 + 1
    xdim, ydim = feature_map_size
    
    gt_box_tensor = make_empty_box_tensor(box_shapes, box_lists, xdim,ydim, num_channels_for_one_box)
    
    for example_ind in range(len(box_lists)):
        gt_boxes = box_lists[example_ind]
        for xind in range(xdim):
            for yind in range(ydim):
                for def_box_ind, box_shape in enumerate(box_shapes):
                    channel_ind = num_channels_for_one_box * def_box_ind
                    xcenter, ycenter = (xind + 0.5) * receptive_field_size, (yind + 0.5) * receptive_field_size
                    width, height = box_shape
                    xmin, xmax, ymin, ymax = _convert_xy_center_wh_box_to_xy_min_max_box(xcenter, ycenter, width, height)
                    if not use_cutoff_boxes:
                        if does_box_get_cutoff(xmin, ymin, xmax, ymax,
                                            frame_xmax=xdim*receptive_field_size, 
                                            frame_ymax=ydim*receptive_field_size):
                            continue

                    default_box = (xcenter, ycenter, width, height)
                    #print default_box
                    is_match, xoff,yoff, scaled_w, scaled_h, cls = match_boxes(default_box, gt_boxes)
                    
                    if is_match:
                        print "hey!"
                        print xoff,yoff, scaled_w, scaled_h
                        # set the correct default box location to be offset from that default box
                        gt_box_tensor[example_ind, channel_ind:channel_ind + 4, xind, yind] = [xoff,yoff, scaled_w, scaled_h]
                        
                        print gt_box_tensor[example_ind, channel_ind:channel_ind + 4, xind, yind]
                        # set class index to class number for categorical encoding
                        gt_box_tensor[example_ind, channel_ind + num_channels_for_one_box, xind, yind] = cls
                
    return gt_box_tensor

