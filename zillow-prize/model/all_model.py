#!/usr/bin/python

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV

params = {
    'objective': 'reg:linear',
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 7,
    'colsample_bytree': 0.2,
    'lambda': 0.6,
    'alpha': 0.8,
    'seed': 400,
    'silent': 1,
    'nthread': 4
}


def turnModel(train_data, params):
    x_train = train_data.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                          'propertycountylandusecode'],
                         axis=1)
    y_train = train_data['logerror'].values

    split = 80000
    x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

    print "Building DMatrix..."

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    print "Training ..."

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)


def turnParams(train_data):
    x_train = train_data.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                               'propertycountylandusecode'],
                              axis=1)
    y_train = train_data['logerror'].values

    params = {
        "objective": "reg:linear",
        "learning_rate": 0.3,
        "max_depth": 6,
        'min_child_weight': 7,
        "seed": 400,
    }

    bst = XGBClassifier(params)

    print "grid search params"
    params_test ={
        "n_estimators": range(400, 600, 100),
        #"max_depth": range(4, 7, 1)
    }

    clf = GridSearchCV(estimator=bst, param_grid=params_test, scoring='accuracy', cv=3)
    clf.fit(x_train, y_train)
    print clf.grid_scores_, clf.best_params_
    print clf.best_score_

if __name__ == "__main__":
    print('Loading data ...')
    train = pd.read_csv('../data/feature_train.csv')

    turnParams(train)
    #turnModel(train, params)
