# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

print "load data..."
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print "train data set:", train.shape
print "test date set:", test.shape

print "clean data..."


def clean_data(titanic):
    titanic["child"] = (titanic["Age"] < 14)

    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    titanic.loc[titanic["Embarked"] == 'S', "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == 'C', "Embarked"] = 2
    titanic.loc[titanic["Embarked"] == 'Q', "Embarked"] = 3
    titanic.loc[titanic["Embarked"].isnull()] = 1

    titanic["family_num"] = titanic["SibSp"] + titanic["Parch"]

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic

train_data = clean_data(train)
test_data = clean_data(test)

# Engineer Features
print "Engineer Feature..."
#predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "child",
#              "family_num"]
predictors = ["Pclass", "Sex","Age", "SibSp", "Parch", "Fare", "Embarked", "child", "family_num"]
print "features:", predictors
# Model of Random Forest
print "fit model..."
rf = RandomForestClassifier(
    #n_estimators=200,
    max_depth=4,
    min_samples_split=6,
    min_samples_leaf=6,
    oob_score=True
)
param_test = {
    'n_estimators': range(150, 220, 10)
    #'min_samples_split': range(5, 10, 1),
    #'min_samples_leaf': range(4, 10, 2)
    #'max_depth': range(2, 8, 2)
}
clf = GridSearchCV(estimator=rf, param_grid=param_test, scoring='accuracy', cv=10)
clf.fit(train_data[predictors], train_data["Survived"])

print(clf.grid_scores_, clf.best_params_, clf.best_score_)

predict_data = clf.predict(test_data[predictors])

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predict_data
})

submission.to_csv('data/random-forest.csv', index=False)
