#!/usr/bin/python

import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


def getEstimator(alg, x_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=150):
    if useTrainCV:
        xgb_params = alg.get_xgb_params()
        xgb_train = xgb.DMatrix(x_train, label=y_train)
        print xgb_params
        cv_result = xgb.cv(xgb_params, xgb_train, num_boost_round=alg.get_params()['n_estimators'],
                           nfold=cv_folds, early_stopping_rounds=early_stopping_rounds)
        n_estimators = cv_result.shape[0]
        print "cv_result is :", cv_result
        print "n_estimator is :", n_estimators
    # Fit the algorithm on the data
    return n_estimators

if __name__ == "__main__":
    print('Loading data ...')
    train = pd.read_csv('../../data/train.csv')
    xgb_params = {
        'objective': 'reg:linear',
        #'base_score': 0,
        'min_child_weight': 12,
        'learning_rate': 0.3,
        'max_depth': 6,
        'n_estimators': 7,
        'seed': 200,
        'silent': 0,
    }
    # Set model
    alg = XGBClassifier(**xgb_params)
    # train data frame
    x_train = train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                          'propertycountylandusecode'],
                         axis=1)
    y_train = train['logerror'].values
    # There n_estimators from xgb.cv
    #n_estimators = getEstimator(alg, x_train, y_train)
    param_test = {
        'max_depth': range(4, 8, 1),
        'n_estimators': [50, 500],

    }
    clf = GridSearchCV(estimator=alg, param_grid=param_test, cv=5, scoring='mean_absolute_error', verbose=1)
    clf.fit(x_train, y_train)
    print "grid search end..."
    print clf.grid_scores_, clf.best_params_, clf.best_score_



