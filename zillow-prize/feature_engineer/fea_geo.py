# -*- encoding:utf-8 -*-
# !/usr/bin/python

import numpy as np
import pandas as pd
import geohash
from fea_util import load_train_data


def cluster_geo(geo_df):
    return 0

def getGeo(lat, lon):
    return geohash.encode(lat, lon)


if __name__ == "__main__":
    train_df = load_train_data()
    geo_df = train_df[['latitude', 'longitude','logerror']]
    geo_df['longitude'] /= 1e6
    geo_df['latitude'] /= 1e6

    geo_df['geohash'] = map(getGeo, geo_df['latitude'], geo_df['longitude'])
    print geo_df.head(5)



