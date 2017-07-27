import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

print "load data..."
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print "train data set:", train.shape
print "test date set:", test.shape

print "clean data..."


def clean_data(titanic):
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    titanic.loc[titanic["Embarked"] == 'S', "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == 'C', "Embarked"] = 2
    titanic.loc[titanic["Embarked"] == 'Q', "Embarked"] = 3
    titanic.loc[titanic["Embarked"].isnull()] = 1

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic

train_data = clean_data(train)
test_data = clean_data(train)

# Engineer Features
print "Engineer Feature..."
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Model of Random Forest
print "fit model..."
rf = RandomForestClassifier(
    n_estimators=100,
    min_samples_split=4,
    min_samples_leaf=2,
    oob_score=True
)

rf.fit(train_data[predictors], train_data["Survived"])
print rf.oob_score

predict_data = rf.predict(test_data[predictors])



