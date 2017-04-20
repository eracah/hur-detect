


import numpy as np
import sys
from convert_between_xy_minmax_xy_center_boxes import _convert_xy_center_wh_box_to_xy_min_max_box



def get_iou(box1,box2):
        #expects box coords of type x_center, y_center, w, h
        x1,y1,w1,h1 = box1[:4]
        x2,y2,w2,h2 = box2[:4]
        xmin1, xmax1, ymin1, ymax1 = _convert_xy_center_wh_box_to_xy_min_max_box(x1,y1,w1,h1, xmax_val=768, ymax_val=768)
        xmin2, xmax2, ymin2, ymax2 = _convert_xy_center_wh_box_to_xy_min_max_box(x2,y2,w2,h2, xmax_val=768, ymax_val=768)

        
        def get_intersection(xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2 ):
            
            inters_width = max(0, min(xmax1, xmax2) - max(xmin1,xmin2))
            
            inters_height = max(0, min(ymax1,ymax2) - max(ymin1,ymin2))
            intersection = inters_width * inters_height
            if intersection < 0:
                print "intersection < 0, at %8.4f so: " %(intersection), xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2
            return intersection
        
        intersection = get_intersection(xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2 )
                         
        def get_area(box_mm):
            xmin, xmax, ymin, ymax = box_mm
            area = (xmax - xmin) * (ymax - ymin)
            return area
                         
        area1 = get_area((xmin1, xmax1, ymin1, ymax1))
        area2 = get_area((xmin2, xmax2, ymin2, ymax2))
        union = area1 + area2 - intersection                                                             

#         print " area1: ", area1, " area2: ", area2, " intersection: ", intersection, " union: ", union
        iou = intersection / float(union)
        return iou





