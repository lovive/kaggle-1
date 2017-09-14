# -*- encoding:utf-8 -*-
# !/usr/bin/python

import numpy as np
import pandas as pd


def load_train_data():
    # load train data
    train = pd.read_csv('../data/train_2016_v2.csv')
    properties = pd.read_csv('../data/properties_2016.csv')
    #sample = pd.read_csv('../data/sample_submission.csv')

    for c, dtype in zip(properties.columns, properties.dtypes):
        if dtype == np.float64:
            properties[c] = properties[c].astype(np.float32)

    train = train.merge(properties, on='parcelid', how='left')

    return train


def load_test_data():
    # load test data
    #train = pd.read_csv('../data/train_2016_v2.csv')
    properties = pd.read_csv('../data/properties_2016.csv')
    sample = pd.read_csv('../data/sample_submission.csv')

    for c, dtype in zip(properties.columns, properties.dtypes):
        if dtype == np.float64:
            properties[c] = properties[c].astype(np.float32)

    sample['parcelid'] = sample['ParcelId']
    test = sample.merge(properties, on='parcelid', how='left')

    return test