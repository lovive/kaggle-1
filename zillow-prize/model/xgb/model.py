#!/usr/bin/python

import numpy as np
import pandas as pd
import xgboost as xgb
import gc
from datetime import datetime
from xgboost import XGBClassifier

# default xgb params
xgb_params = {
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    # 'base_score': 0,
    'min_child_weight': 12,
    'learning_rate': 0.003,
    'max_depth': 8,
    'subsample': 0.8,
    'lambda': 0.8,
    'alpha': 0.4,
    'base_score': 0,
    'seed': 500,
    'silent': 1,
}


# split train_df to train and valid
def split_turn_model(x_train_df, y_train_df):
    split = 80000
    x_train, y_train, x_valid, y_valid = x_train_df[:split], y_train_df[:split], x_train_df[split:], y_train_df[split:]

    xgb_train = xgb.DMatrix(x_train, label=y_train)
    xgb_valid = xgb.DMatrix(x_valid, label=y_valid)
    del x_train, y_train;
    gc.collect()

    watchlist = [(xgb_train, 'train'), (xgb_valid, 'valid')]
    clf = xgb.train(xgb_params, xgb_train, 10000, watchlist, early_stopping_rounds=50, verbose_eval=1)
    del xgb_train, xgb_valid; gc.collect()

    return clf


def cv_turn_model(x_train, y_train):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    # cross-validation
    cv_result = xgb.cv(xgb_params,
                       dtrain,
                       nfold=5,
                       num_boost_round=5000,
                       early_stopping_rounds=50,
                       verbose_eval=1,
                       show_stdv=False
                       )
    num_boost_rounds = len(cv_result)
    print num_boost_rounds

    # train model
    clf = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds)

    return clf


def submision(alg, feature):
    tests = pd.read_csv('../../data/feature_test.csv', chunksize=100000, iterator=True)
    pre = []
    for test in tests:
        dtest = xgb.DMatrix(test[features])
        del test; gc.collect()

        pre_test = alg.predict(dtest)
        pre.extend(pre_test)
        print "predict: %s..." % len(pre)
        del dtest; gc.collect()

    print "submit..."
    sub = pd.read_csv('../../data/sample_submission.csv')
    for c in sub.columns[sub.columns != 'ParcelId']:
        sub[c] = pre
    submit_file = '../../sub/{}.csv.gz'.format(datetime.now().strftime('%Y%m%d_%H_%M'))
    sub.to_csv(submit_file, index=False, compression='gzip')


if __name__ == "__main__":
    print('Loading data ...')
    train = pd.read_csv('../../data/feature_train.csv')
    # drop out ouliers
    train = train[train.logerror > -0.4]
    train = train[train.logerror < 0.42]

    x_train = train.drop(['parcelid', 'logerror','transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
    y_train = train["logerror"].values

    features = x_train.columns

    del train; gc.collect()

    print "train model 1"
    #xgb1 = split_turn_model(x_train, y_train)
    #submision(xgb1, features)

    print "train model 2"
    xgb2 = cv_turn_model(x_train, y_train)
    submision(xgb2, features)


