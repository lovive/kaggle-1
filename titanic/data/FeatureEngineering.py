# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

print "load data..."
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

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

train = clean_data(train)
test = clean_data(test)

# Feature Engineering
def FeatureEngineer(titanic):
    # sex
    titanic["sex"] = titanic["Sex"].apply(lambda x: 1 if x == "male" else 0)

    # child
    titanic["child"] = titanic["Age"].apply(lambda x: 1 if x < 16 else 0)

    # familysize
    titanic["fimalysize"] = titanic["SibSp"] + titanic["Parch"]

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
    # Name
    def getName(name):
        if "Mr" in str(name):
            return 1
        elif "Mrs" in str(name):
            return 2
        else:
            return 0

    train["name"] = train["Name"].apply(getName)

    return titanic

train = FeatureEngineer(train)
test = FeatureEngineer(test)

train.to_csv("feature_train.csv", encoding="utf-8")


