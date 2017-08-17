# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

print "load data..."
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print "clean data..."


def clean_data(titanic):
    # chile
    titanic["child"] = (titanic["Age"] < 16)

    # Age
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    # Sex
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    # Embarked
    titanic.loc[titanic["Embarked"] == 'S', "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == 'C', "Embarked"] = 2
    titanic.loc[titanic["Embarked"] == 'Q', "Embarked"] = 3
    titanic.loc[titanic["Embarked"].isnull()] = 0

    # Name
    def replace_name(name):
        if "Mr" in str(name):
            return "1"
        elif "Mrs" in str(name):
            return "2"
        else:
            return "0"
    titanic["name"] = titanic["Name"].apply(replace_name).astype(int)

    # Carbin
    def replace_cabin(cabin):
        return str(cabin)[:1]
    titanic["Cabin"] = titanic["Cabin"].apply(replace_cabin)

    # family_num
    titanic["family_num"] = titanic["SibSp"] + titanic["Parch"]

    # Fare
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic

train_data = clean_data(train)
test_data = clean_data(test)

# Engineer Features
print "Engineer Feature..."

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "child",
              "family_num", "name"]

# Model of Random Forest
print "fit model..."
rf = RandomForestClassifier(
    #n_estimators=150,
    #random_state=1,
    #max_depth=4,
    #min_samples_split=5,
    #min_samples_leaf=4,
    n_jobs=-1,
    oob_score=True
)
param_test = {
    'n_estimators': range(140, 170, 10),
    'min_samples_split': range(5, 7, 1),
    'min_samples_leaf': range(4, 7, 1),
    'max_depth': range(3, 6, 1)
}
clf = GridSearchCV(estimator=rf, param_grid=param_test, scoring='accuracy', cv=10)
clf.fit(train_data[predictors], train_data["Survived"])

print(clf.best_params_, clf.best_score_)

predict_data = clf.predict(test_data[predictors])

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predict_data
})

submission.to_csv('data/random-forest.csv', index=False)
