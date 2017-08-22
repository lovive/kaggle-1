# -*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score


def submit(alg, test_data):
    predict_data = alg.predict(test_data[features])

    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predict_data
    })

    submission.to_csv('../data/bagging.csv', index=False)

if __name__ == '__main__':
    train = pd.read_csv('../data/feature_train.csv')
    test = pd.read_csv('../data/feature_test.csv')
    print train.head()

    features = ["Pclass", "Fare", "Age", "SibSp", "Parch", "sex", "child", "fimalysize", "embark", "name", "cabin"]
    x_train = train[features]
    y_train = train["Survived"]

    rf = RandomForestClassifier(n_estimators=140, max_depth=4, min_samples_split=6, min_samples_leaf=4, n_jobs=4)
    rf_scores = cross_val_score(rf, x_train, y_train, cv=3)

    dt = DecisionTreeClassifier()
    dt_scores = cross_val_score(dt, x_train, y_train)

    bagging_clf = BaggingClassifier(rf, max_samples=0.9, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=4)
    bagging_clf.fit(x_train, y_train)
    bagging_scores = cross_val_score(bagging_clf, x_train, y_train, cv=3)

    print "rf scores:", rf_scores
    print "dt scores:", dt_scores
    print "bagging scores:", bagging_scores

    submit(bagging_clf, test)