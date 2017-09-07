# -*- encoding:utf-8 -*-
# !/usr/bin/python

import numpy as np
import pandas as pd
import datetime as dt

# load data
def load_data():
    train = pd.read_csv('../data/train_2016_v2.csv')
    properties = pd.read_csv('../data/properties_2016.csv')
    sample = pd.read_csv('../data/sample_submission.csv')

    return train, properties, sample

# build to float32 to save memory
def clean_data(dataframe):
    print "Binding to float32"
    for c, dtype in zip(dataframe.columns, dataframe.dtypes):
        if dtype == np.float64:
            dataframe[c] = dataframe[c].astype(np.float32)

    return dataframe

# Feature Engineering
def feature_select(dataframe):
    print "Feature engineering..."

    # add month feature
    dataframe["transactiondate"] = pd.to_datetime(dataframe["transactiondate"])
    dataframe["Month"] = dataframe["transactiondate"].dt.month

    print "Step1:change object to bool"
    for c in dataframe.dtypes[dataframe.dtypes == object].index.values:
        dataframe[c] = (dataframe[c] == True)
    print "Step2:add feature"

    # error in calculation of the finished living area of home
    dataframe['N-LivingAreaError'] = dataframe['calculatedfinishedsquarefeet'] / dataframe[
        'finishedsquarefeet12']

    # add null of calculatedfinishedsquarefeet
    dataframe['finishedsquarefeet12'] = dataframe['finishedsquarefeet12'].fillna(dataframe['calculatedfinishedsquarefeet'])

    # proportion of living area
    dataframe['N-LivingAreaProp'] = dataframe['calculatedfinishedsquarefeet'] / dataframe[
        'lotsizesquarefeet']
    dataframe['N-LivingAreaProp2'] = dataframe['finishedsquarefeet12'] / dataframe[
        'finishedsquarefeet15']
    # Amout of extra space
    dataframe['N-ExtraSpace'] = dataframe['lotsizesquarefeet'] - dataframe[
        'calculatedfinishedsquarefeet']
    dataframe['N-ExtraSpace-2'] = dataframe['finishedsquarefeet15'] - dataframe[
        'finishedsquarefeet12']

    # Total number of rooms
    dataframe['N-TotalRooms'] = dataframe['bathroomcnt'] + dataframe['bedroomcnt']
    # Average room size
    dataframe['N-AvRoomSize'] = dataframe['calculatedfinishedsquarefeet'] / dataframe['roomcnt']

    # Number of Extra rooms
    dataframe['N-ExtraRooms'] = dataframe['roomcnt'] - dataframe['N-TotalRooms']
    # Ratio of the built structure value to land area
    dataframe['N-ValueProp'] = dataframe['structuretaxvaluedollarcnt'] / dataframe[
        'landtaxvaluedollarcnt']

    # Does property have a garage, pool or hot tub and AC?
    dataframe['N-GarPoolAC'] = ((dataframe['garagecarcnt'] > 0) & (dataframe['pooltypeid10'] > 0) & (
        dataframe['airconditioningtypeid'] != 5)) * 1

    dataframe["N-location"] = dataframe["latitude"] + dataframe["longitude"]
    dataframe["N-location-2"] = dataframe["latitude"] * dataframe["longitude"]
    dataframe["N-location-2round"] = dataframe["N-location-2"].round(-4)

    dataframe["N-latitude-round"] = dataframe["latitude"].round(-4)
    dataframe["N-longitude-round"] = dataframe["longitude"].round(-4)

    # Ratio of tax of property over parcel
    dataframe['N-ValueRatio'] = dataframe['taxvaluedollarcnt'] / dataframe['taxamount']

    # TotalTaxScore
    dataframe['N-TaxScore'] = dataframe['taxvaluedollarcnt'] * dataframe['taxamount']

    # polnomials of tax delinquency year
    dataframe["N-taxdelinquencyyear-2"] = dataframe["taxdelinquencyyear"] ** 2
    dataframe["N-taxdelinquencyyear-3"] = dataframe["taxdelinquencyyear"] ** 3

    # Length of time since unpaid taxes
    dataframe['N-life'] = 2018 - dataframe['taxdelinquencyyear']

    # Number of properties in the zip
    zip_count = dataframe['regionidzip'].value_counts().to_dict()
    dataframe['N-zip_count'] = dataframe['regionidzip'].map(zip_count)

    # Number of properties in the city
    city_count = dataframe['regionidcity'].value_counts().to_dict()
    dataframe['N-city_count'] = dataframe['regionidcity'].map(city_count)

    # Number of properties in the city
    region_count = dataframe['regionidcounty'].value_counts().to_dict()
    dataframe['N-county_count'] = dataframe['regionidcounty'].map(region_count)

    # Indicator whether it has AC or not
    dataframe['N-ACInd'] = (dataframe['airconditioningtypeid'] != 5) * 1

    # Indicator whether it has Heating or not
    dataframe['N-HeatInd'] = (dataframe['heatingorsystemtypeid'] != 13) * 1
    # polnomials of the variable
    dataframe["N-structuretaxvaluedollarcnt-2"] = dataframe["structuretaxvaluedollarcnt"] ** 2
    dataframe["N-structuretaxvaluedollarcnt-3"] = dataframe["structuretaxvaluedollarcnt"] ** 3

    # Average structuretaxvaluedollarcnt by city
    group = dataframe.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
    dataframe['N-Avg-structuretaxvaluedollarcnt'] = dataframe['regionidcity'].map(group)

    # Deviation away from average
    dataframe['N-Dev-structuretaxvaluedollarcnt'] = abs(
        (dataframe['structuretaxvaluedollarcnt'] - dataframe['N-Avg-structuretaxvaluedollarcnt'])) / dataframe[
                                                       'N-Avg-structuretaxvaluedollarcnt']
    return dataframe

def feature_engineer(dataframe):
    # clean data
    dataframe = clean_data(dataframe)
    # feature select
    dataframe = feature_select(dataframe)

    return dataframe

if __name__ == "__main__":
    print "Loading data ..."
    train, properties, sample = load_data()

    train = train.merge(properties, on='parcelid', how='left')

    sample['parcelid'] = sample['ParcelId']
    test = sample.merge(properties, on='parcelid', how='left')
    ####add month feature assuming 2016-10-01
    test["transactiondate"] = '2016-07-01'

    train = feature_engineer(train)
    test = feature_engineer(test)

    print train.info()

    print "write to csv: feature_train.csv and feature_test.csv"
    train.to_csv("../data/feature_train.csv", encoding="utf-8")
    test.to_csv("../data/feature_test.csv", encoding="utf-8")