# -*- encoding:utf-8 -*-
# !/usr/bin/python

import numpy as np
import pandas as pd


class FeatureEngineering():
    def __init__(self):
        self.train_file = '../data/train_2016.csv'
        self.properties_file = '../data/properties_2016.csv'
        self.sample_file = '../data/sample_submission.csv'

    def load_data(self):
        train = pd.read_csv(self.train_file)
        properties = pd.read_csv(self.properties_file)
        sample = pd.read_csv(self.sample_file)

        return train, properties, sample


    def clean_data(self, DataFrame):
        print "Binding to float32"
        for c, dtype in zip(DataFrame.columns, DataFrame.dtypes):
            if dtype == np.float64:
                DataFrame[c] = DataFrame[c].astype(np.float32)
        return DataFrame


    def feature_engineer(self, CleanDataframe):
        print "change object to bool"
        for c in CleanDataframe.dtypes[CleanDataframe.dtypes == object].index.values:
            CleanDataframe[c] = (CleanDataframe[c] == True)

        # dataframe = dataframe.drop(['transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
        CleanDataframe['N-life'] = 2018 - CleanDataframe['yearbuilt']

        # error in calculation of the finished living area of home
        CleanDataframe['N-LivingAreaError'] = CleanDataframe['calculatedfinishedsquarefeet'] / CleanDataframe['finishedsquarefeet12']
        # proportion of living area
        CleanDataframe['N-LivingAreaProp'] = CleanDataframe['calculatedfinishedsquarefeet'] / CleanDataframe['lotsizesquarefeet']
        CleanDataframe['N-LivingAreaProp2'] = CleanDataframe['finishedsquarefeet12'] / CleanDataframe['finishedsquarefeet15']
        # Amout of extra space
        CleanDataframe['N-ExtraSpace'] = CleanDataframe['lotsizesquarefeet'] - CleanDataframe['calculatedfinishedsquarefeet']
        CleanDataframe['N-ExtraSpace-2'] = CleanDataframe['finishedsquarefeet15'] - CleanDataframe['finishedsquarefeet12']

        # Total number of rooms
        CleanDataframe['N-TotalRooms'] = CleanDataframe['bathroomcnt'] + CleanDataframe['bedroomcnt']

        # Average room size
        CleanDataframe['N-AvRoomSize'] = CleanDataframe['calculatedfinishedsquarefeet'] / CleanDataframe['roomcnt']

        # Number of Extra rooms
        CleanDataframe['N-ExtraRooms'] = CleanDataframe['roomcnt'] - CleanDataframe['N-TotalRooms']

        # Ratio of the built structure value to land area
        CleanDataframe['N-ValueProp'] = CleanDataframe['structuretaxvaluedollarcnt'] / CleanDataframe['landtaxvaluedollarcnt']

        # Does property have a garage, pool or hot tub and AC?
        CleanDataframe['N-GarPoolAC'] = ((CleanDataframe['garagecarcnt'] > 0) & (CleanDataframe['pooltypeid10'] > 0) & (
        CleanDataframe['airconditioningtypeid'] != 5)) * 1

        CleanDataframe["N-location"] = CleanDataframe["latitude"] + CleanDataframe["longitude"]
        CleanDataframe["N-location-2"] = CleanDataframe["latitude"] * CleanDataframe["longitude"]
        CleanDataframe["N-location-2round"] = CleanDataframe["N-location-2"].round(-4)

        CleanDataframe["N-latitude-round"] = CleanDataframe["latitude"].round(-4)
        CleanDataframe["N-longitude-round"] = CleanDataframe["longitude"].round(-4)

        return CleanDataframe

    def get_features(self):
        # load data
        train, properties, sample = self.load_data()

        properties = self.clean_data(properties)

        # merge properties with train and sample
        train = train.merge(properties, on='parcelid', how='left')

        sample['parcelid'] = sample['ParcelId']
        test = sample.merge(properties, on='parcelid', how='left')

        feature_train = self.feature_engineer(train)
        feature_test = self.feature_engineer(test)

        clean_train = self.clean_data(train)
        clean_test = self.clean_data(test)

        feature_train = self.feature_engineer(train)
        feature_test = self.feature_engineer(test)

        return feature_train, feature_test

if __name__ == "__main__":
    feature_engineer = FeatureEngineering()
    train_df, test_df = feature_engineer.get_features()

    print "write to csv: data/feature_train.csv and data/feature_test.csv"

    train_df.to_csv("../data/feature_train.csv", encoding="utf-8")
    test_df.to_csv("../data/feature_test.csv", encoding="utf-8")

    print test_df.info()

    #train_df.to_csv("../data/feature_train.csv", encoding="utf-8")

    print "Feature engineering end!"
