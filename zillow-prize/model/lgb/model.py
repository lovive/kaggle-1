#!/usr/bin/python

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import gc
from datetime import datetime
from xgboost import XGBClassifier

# default lgb params
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'          # or 'mae'
params['sub_feature'] = 0.345    # feature_fraction (small values => use very different submodels)
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3


# split train_df to train and valid
def split_turn_model(x_train_df, y_train_df):
    split = 80000
    x_train, y_train, x_valid, y_valid = x_train_df[:split], y_train_df[:split], x_train_df[split:], y_train_df[split:]

    lgb_train = lgb.Dataset(x_train, label=y_train)
    lgb_valid = lgb.Dataset(x_valid, label=y_valid)
    del x_train, y_train;
    gc.collect()

    watchlist = [lgb_valid]
    #watchlist = [(lgb_train, 'train'), (lgb_valid, 'valid')]
    clf = lgb.train(params, lgb_train, 2000, valid_sets=watchlist, early_stopping_rounds=50)
    del lgb_train, lgb_valid; gc.collect()

    return clf


def submision(alg, feature):
    tests = pd.read_csv('../../data/feature_test.csv', chunksize=100000, iterator=True)
    pre = []
    for test in tests:
        dtest = test[features]
        del test; gc.collect()

        pre_test = alg.predict(dtest)
        pre.extend(pre_test)
        print "predict: %s..." % len(pre)
        del dtest; gc.collect()

    print "submit..."
    #pre = 0.97 * pre + 0.03 * 0.011
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
    lgb1 = split_turn_model(x_train, y_train)
    #submision(lgb1, features)


