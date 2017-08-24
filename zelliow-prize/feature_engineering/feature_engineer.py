# -*- encoding:utf-8 -*-
# !/usr/bin/python

import numpy as np
import pandas as pd


# load data
def loadData():
    train = pd.read_csv('../data/train_2016.csv')
    properties = pd.read_csv('../data/properties_2016.csv')
    sample = pd.read_csv('../data/sample_submission.csv')

    return train, properties, sample


# build to float32 to save memory
def cleanData(dataframe):
    print "Binding to float32"
    for c, dtype in zip(dataframe.columns, dataframe.dtypes):
        if dtype == np.float64:
            dataframe[c] = dataframe[c].astype(np.float32)

    return dataframe


# Feature Engineering
def featureSelect(dataframe):
    print "Feature engineering..."

    print "change object to bool"
    for c in dataframe.dtypes[dataframe.dtypes == object].index.values:
        dataframe[c] = (dataframe[c] == True)
    #dataframe = dataframe.drop(['transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)

    return dataframe

if __name__ == "__main__":
    print "Loading data ..."
    train, properties, sample = loadData()
    properties = cleanData(properties)

    train = train.merge(properties, on='parcelid', how='left')

    sample['parcelid'] = sample['ParcelId']
    #test = sample.merge(properties, on='parcelid', how='left')

    train = featureSelect(train)
    #test = featureSelect(test)

    print train.info()

    print "write to csv: feature_train.csv and feature_test.csv"
    #train.to_csv("../data/feature_train.csv", encoding="utf-8")
    #test.to_csv("../data/feature_test.csv", encoding="utf-8")