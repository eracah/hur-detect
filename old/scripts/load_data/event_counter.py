
import matplotlib; matplotlib.use("agg")






def count_events(year):
    metadata_dir = "/storeSSD/eracah/data/metadata/"
    coord_keys = ["xmin", "xmax", "ymin", "ymax"]
    d ={"us-ar":0, "etc":0, "tc":0}
    for weather_type in d.keys(): 
        if weather_type == 'us-ar':
            labeldf = pd.read_csv(join(metadata_dir, 'ar_labels.csv'))
            labeldf = labeldf.ix[(labeldf.year==year)]
            d[weather_type] = len(labeldf)
#             tmplabeldf=labeldf.ix[ (labeldf.month==ts.month) & (labeldf.day==ts.day) & (labeldf.year==ts.year) ].copy()
        else:
            labeldf = pd.read_csv(join(metadata_dir, '_'.join([str(year),weather_type, 'labels.csv'])))
#             tmplabeldf=labeldf.ix[ (labeldf.month==ts.month) & (labeldf.day==ts.day) ].copy()
            if weather_type == "etc":
                d[weather_type] = len(labeldf)
            else:
                d["td"] = len(labeldf[labeldf["str_category"] == "tropical_depression"])
                d["tc"] = len(labeldf[labeldf["str_category"] == "tropical_cyclone"])
            
    return d

def count_events_md(year, last_md = (3,16)):
    lm, ld = last_md
    metadata_dir = "/storeSSD/eracah/data/metadata/"
    coord_keys = ["xmin", "xmax", "ymin", "ymax"]
    d ={"us-ar":0, "etc":0, "tc":0}
    for weather_type in d.keys(): 
        if weather_type == 'us-ar':
            labeldf = pd.read_csv(join(metadata_dir, 'ar_labels.csv'))
            for i in range(1,lm):
                labeldf = labeldf.ix[(labeldf.year==year) & (labeldf.month == i)]
                d[weather_type] += len(labeldf)
            for i in range(1,ld+1):
                labeldf_end = labeldf.ix[(labeldf.year==year) & (labeldf.month ==lm) & (labeldf.day == i)]
                d[weather_type] += len(labeldf_end)
           
        else:
            labeldf = pd.read_csv(join(metadata_dir, '_'.join([str(year),weather_type, 'labels.csv'])))
            if weather_type == "etc":
                for i in range(1,lm):
                    labeldfi = labeldf.ix[(labeldf.year==year) & (labeldf.month == i)]
                    d[weather_type] += len(labeldfi)
                for i in range(1,ld+1):
                    labeldfi = labeldf.ix[(labeldf.year==year) & (labeldf.month ==lm) & (labeldf.day == i)]
                    d[weather_type] += len(labeldfi)
            else:
                d["td"] = 0
                for i in range(1,lm):
                    labeldfi = labeldf.ix[(labeldf.year==year) & (labeldf.month == i)]
                    d["td"] += len(labeldfi[labeldfi["str_category"] == "tropical_depression"])
                    d["tc"] += len(labeldfi[labeldfi["str_category"] == "tropical_cyclone"])
                for i in range(1,ld+1):
                    labeldfi = labeldf.ix[(labeldf.year==year) & (labeldf.month ==lm) & (labeldf.day == i)]
                    d["td"] += len(labeldfi[labeldfi["str_category"] == "tropical_depression"])
                    d["tc"] += len(labeldfi[labeldfi["str_category"] == "tropical_cyclone"])
              
    return d

def get_percents(event_dict):
    tot = sum(event_dict.values())
    new_d = {k: float(v)/ tot for k,v in event_dict.iteritems()}
    return new_d

