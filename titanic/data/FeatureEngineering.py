# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


# First step: Cleaning Data
def clean_data(titanic):
    # Age
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    # Fare
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    # Cabin
    titanic["Cabin"] = titanic["Cabin"].fillna("N")

    # Embarked
    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    return titanic

# Feature Engineering
def FeatureEngineer(titanic):
    # sex
    titanic["sex"] = titanic["Sex"].apply(lambda x: 1 if x == "male" else 0)

    # child
    titanic["child"] = titanic["Age"].apply(lambda x: 1 if x < 16 else 0)

    # familysize
    titanic["fimalysize"] = titanic["SibSp"] + titanic["Parch"] + 1

    # embark
    def getEmbark(Embarked):
        if Embarked == "S":
            return 1
        elif Embarked == "C":
            return 2
        else:
            return 3
    titanic["embark"] = titanic["Embarked"].apply(getEmbark)

    # name
    def getName(name):
        if "Mr" in str(name):
            return 1
        elif "Mrs" in str(name):
            return 2
        else:
            return 0
    titanic["name"] = titanic["Name"].apply(getName)

    # cabin
    def getCabin(cabin):
        if cabin == "N":
            return 0
        else:
            return 1
    titanic["cabin"] = titanic["Cabin"].apply(getCabin)

    return titanic

if __name__ == "__main__":
    print "load data..."
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    print "clean data..."
    train = clean_data(train)
    test = clean_data(test)

    print "feature engineering..."
    train = FeatureEngineer(train)
    test = FeatureEngineer(test)

    print test.head(3)

    print "write to csv: feature_train.csv and feature_test.csv"
    train.to_csv("feature_train.csv", encoding="utf-8")
    test.to_csv("feature_test.csv", encoding="utf-8")


