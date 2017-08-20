#!/usr/bin/python

import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


def modelfit(alg, x_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_params = alg.get_xgb_params()

        xgb_train = xgb.DMatrix(x_train, label=y_train)

        print xgb_params

        cv_result = xgb.cv(xgb_params, xgb_train, num_boost_round=alg.get_params()['n_estimators'],
                           nfold=cv_folds, metrics='mae', early_stopping_rounds=early_stopping_rounds)

        n_estimators = cv_result.shape[0]

        alg.set_params(n_estimators=n_estimators)

        print cv_result

    # Fit the algorithm on the data
    alg.fit(x_train, y_train, eval_metric='mae')

    # Predict training set:
    train_predprob = alg.predict_proba(x_train)
    #logloss = log_loss(y_train, train_predprob)

    # Print model report:
    #print ("logloss of train :")
    #print logloss


if __name__ == "__main__":
    print('Loading data ...')
    train = pd.read_csv('../../data/feature_train.csv')

    # Set model
    xgb1 = XGBClassifier(
        objective="reg:linear",
        max_depth=6,
        #eval_metric="mae",
        n_estimators=10000,
        learning_rate=0.3,

        seed=400,
        nthread=4,
        silent=0
    )
    # train data frame
    x_train = train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                          'propertycountylandusecode'],
                         axis=1)
    y_train = train['logerror'].values

    #turnParams(train)
    #turnModel(train, params)
    modelfit(xgb1, x_train, y_train)



