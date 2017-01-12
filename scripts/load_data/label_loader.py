
import matplotlib; matplotlib.use("agg")


import sys
from util import get_timestamp
import pandas as pd
import numpy as np
from os.path import join



def match_nc_to_csv(fname, weather_type,metadata_dir, inc_csv=False):
        coord_keys = ["xmin", "xmax", "ymin", "ymax"]
        ts=get_timestamp(fname)

        if weather_type == 'us-ar':
            labeldf = pd.read_csv(join(metadata_dir, 'ar_labels.csv'))
            tmplabeldf=labeldf.ix[ (labeldf.month==ts.month) & (labeldf.day==ts.day) & (labeldf.year==ts.year) ].copy()
        else:
            labeldf = pd.read_csv(join(metadata_dir, '_'.join([str(ts.year),weather_type, 'labels.csv'])))
            tmplabeldf=labeldf.ix[ (labeldf.month==ts.month) & (labeldf.day==ts.day) ].copy()


        selectdf=tmplabeldf[["time_step"]+ coord_keys + ["category"]]
        if inc_csv is True:
            return selectdf, labeldf
        else:
            return selectdf 



def make_labels_for_dataset(fname, kwargs):
    '''takes in string for fname and the number of time_steps and outputs
    a time_steps by maximages by 5 tensor encoding the coordinates and class of each event in a time step'''

    weather_types = ['tc','etc', 'us-ar']
    ts=get_timestamp(fname)
    maximagespertimestep=25
    time_steps_per_file, metadata_dir = kwargs["time_steps_per_file"], kwargs["metadata_dir"]
    # for every time step for every possible event, xmin,xmax,ymin,ymax,class
    bboxes = np.zeros((time_steps_per_file, maximagespertimestep, 5))
    event_counter = np.zeros((time_steps_per_file,))
    for weather_type in weather_types:
        selectdf = match_nc_to_csv(fname, weather_type, metadata_dir)

        timelist=set(selectdf["time_step"])
        for t in timelist:
            t = int(t)

            coords_for_t = selectdf[selectdf["time_step"]==t].drop(["time_step"], axis=1).values
            coords_for_t = coords_for_t[(coords_for_t > 0).all(1)]

            # get current number of events and number of events for this time step
            num_events_for_t = coords_for_t.shape[0]
            cur_num_events = int(event_counter[t])

            #make slice
            slice_for_t = slice(cur_num_events, cur_num_events + num_events_for_t)

            #fill variables
            bboxes[t, slice_for_t] = coords_for_t
            event_counter[t] += num_events_for_t
    return bboxes



sys.path.insert(0,"/home/evan/hur-detect/scripts/")
from configs import *
kwargs = process_kwargs(save_res=False)



make_labels_for_dataset("cam5_1_amip_run2.cam2.h2.1979-01-17-00000.nc",kwargs)





