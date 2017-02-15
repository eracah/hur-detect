
import matplotlib; matplotlib.use("agg")


import sys
from util import get_timestamp
import pandas as pd
import numpy as np
from os.path import join
import os



def get_cols(df):
    des_keys = ["xmin", "xmax", "ymin", "ymax",
               "time_step", "category", "month", "year", "day", "str_category"]
    df = df[des_keys]
    return df
    

def merge_all_csvs(metadata_dir):
    dfs = []
    for year in range(1979,2006):
        for weather_type in ["tc", "etc"]:
            csv_fn = join(metadata_dir, '_'.join([str(year), weather_type, 'labels.csv']))
            df = pd.read_csv(csv_fn)
            df = get_cols(df)
            dfs.append(df)
    merged_df = pd.concat(dfs)
    merged_df.to_csv(join(metadata_dir, "all_non_ar.csv"))
    return merged_df

def merge_two_csvs(metadata_dir, fn1,fn2):
    fn1 = join(metadata_dir, fn1)
    fn2 = join(metadata_dir, fn2)
    dfs = [pd.read_csv(fn1), pd.read_csv(fn2)]
    final_df = pd.concat(dfs)
    final_df.to_csv(join(metadata_dir, "labels.csv"))

