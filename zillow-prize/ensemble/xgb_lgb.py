#!/usr/bin/python

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import gc

print "========XGBoost start=========="
# xgb 1 params
xgb1_params = {}
xgb1_params['objective'] = 'reg:linear'
xgb1_params['eval_metric'] = 'mae'
xgb1_params['min_child_weight'] = 12
xgb1_params['learning_rate'] = 0.0021
xgb1_params['max_depth'] = 8
xgb1_params['subsample'] = 0.77
xgb1_params['lambda'] = 0.8
xgb1_params['alpha'] = 0.4
xgb1_params['base_score'] = 0
xgb1_params['seed'] = 400
xgb1_params['silent'] = 1

# xgb 1 params
xgb2_params = {}
xgb2_params['objective'] = 'reg:linear'
xgb2_params['eval_metric'] = 'mae'
xgb2_params['min_child_weight'] = 10
xgb2_params['learning_rate'] = 0.003
xgb2_params['max_depth'] = 6
xgb2_params['base_score'] = 0
xgb2_params['seed'] = 400
xgb2_params['silent'] = 1

print('Load train data ...')
train = pd.read_csv('../data/feature_train.csv')
# drop out ouliers
train = train[train.logerror > -0.4]
train = train[train.logerror < 0.42]

x_train = train.drop(['parcelid', 'logerror','transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = train["logerror"].values

x_train_df, x_valid_df, y_train_df, y_valid_df = train_test_split(x_train, y_train, test_size=0.2)

xgb_train = xgb.DMatrix(x_train_df, label=y_train_df)
xgb_valid = xgb.DMatrix(x_valid_df, label=y_valid_df)

watchlist = [(xgb_train, 'train'), (xgb_valid, 'valid')]
xgb1 = xgb.train(xgb1_params, xgb_train, 10000, watchlist, early_stopping_rounds=50, verbose_eval=1)
xgb1_pre = []
tests = pd.read_csv('../data/feature_test.csv', chunksize=100000, iterator=True)
for test in tests:
    dtest = xgb.DMatrix(test[x_train.columns])
    pre_test = xgb1.predict(dtest)
    xgb1_pre.extend(pre_test)
    print "predict: %s..." % len(xgb1_pre)
del test, tests; gc.collect()

xgb2 = xgb.train(xgb2_params, xgb_train, 10000, watchlist, early_stopping_rounds=50, verbose_eval=1)
xgb2_pre = []
tests = pd.read_csv('../data/feature_test.csv', chunksize=100000, iterator=True)
for test in tests:
    dtest = xgb.DMatrix(test[x_train.columns])
    pre_test = xgb2.predict(dtest)
    xgb2_pre.extend(pre_test)
    print "predict: %s..." % len(xgb2_pre)
del test, tests; gc.collect()

print "========LigbtGBM start=========="
# default lgb params
lgb_params = {}
lgb_params['max_bin'] = 10
lgb_params['learning_rate'] = 0.0021 # shrinkage_rate
lgb_params['boosting_type'] = 'gbdt'
lgb_params['objective'] = 'regression'
lgb_params['metric'] = 'mae'          # or 'mae'
lgb_params['sub_feature'] = 0.345    # feature_fraction (small values => use very different submodels)
lgb_params['bagging_fraction'] = 0.85 # sub_row
lgb_params['bagging_freq'] = 40
lgb_params['num_leaves'] = 512        # num_leaf
lgb_params['min_data'] = 500         # min_data_in_leaf
lgb_params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
lgb_params['verbose'] = 0
lgb_params['feature_fraction_seed'] = 2
lgb_params['bagging_seed'] = 3

lgb_train = lgb.Dataset(x_train_df, label=y_train_df)
lgb_valid = lgb.Dataset(x_valid_df, label=y_valid_df)

lgb_watchlist = [lgb_valid]
lgb = lgb.train(lgb_params, lgb_train, 10000, valid_sets=lgb_watchlist, early_stopping_rounds=50)
lgb_pre = []
tests = pd.read_csv('../data/feature_test.csv', chunksize=100000, iterator=True)
for test in tests:
    dtest = test[x_train.columns]
    pre_test = lgb.predict(dtest)
    lgb_pre.extend(pre_test)
    print "predict: %s..." % len(lgb_pre)
del test, tests; gc.collect()
'''
print "========Linear Regression start=========="
reg = LinearRegression(n_jobs=-1)
reg.fit(x_train, y_train)

reg_pre = []
tests = pd.read_csv('../data/feature_test.csv', chunksize=100000, iterator=True)
for test in tests:
    dtest = test[test[x_train.columns]]
    pre_test = lgb.predict(dtest)
    lgb_pre.extend(pre_test)
'''
xgb1_weight = 0.4
xgb2_weight = 0.2

BASELINE_WEIGHT = 0.0056
BASELINE_PRED = 0.0115

lgb_weight = 1 - xgb1_weight

pred = xgb1_weight*np.array(xgb1_pre) + lgb_weight*np.array(lgb_pre) + xgb2_weight*np.array(xgb2_pre)

print "submit..."
sub = pd.read_csv('../data/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = pred
submit_file = '../sub/{}.csv.gz'.format(datetime.now().strftime('%Y%m%d_%H_%M'))
sub.to_csv(submit_file, index=False, compression='gzip')


