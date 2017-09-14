#!/usr/bin/python

import numpy as np
import pandas as pd
import xgboost as xgb
import gc
import matplotlib.pyplot as plt

# default xgb params
xgb_params = {
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    # 'base_score': 0,
    'min_child_weight': 12,
    'learning_rate': 0.03,
    'max_depth': 8,
    'subsample': 0.8,
    'lambda': 0.8,
    'alpha': 0.4,
    'base_score': 0,
    'seed': 500,
    'silent': 1,
}

print('Loading data ...')
train = pd.read_csv('../data/feature_train.csv')

x_train_df = train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'],
                     axis=1)
y_train_df = train["logerror"].values

split = 80000
x_train, y_train, x_valid, y_valid = x_train_df[:split], y_train_df[:split], x_train_df[split:], y_train_df[split:]

xgb_train = xgb.DMatrix(x_train, label=y_train)
xgb_valid = xgb.DMatrix(x_valid, label=y_valid)
del x_train, y_train;
gc.collect()

watchlist = [(xgb_train, 'train'), (xgb_valid, 'valid')]
clf = xgb.train(xgb_params, xgb_train, 10000, watchlist, early_stopping_rounds=50, verbose_eval=1)

del xgb_train, xgb_valid; gc.collect()

fig = plt.subplots(figsize=(16, 9))
#clf.plot_importance()
feat_imp = pd.Series(clf.get_fscore()).sort_values(ascending=False)
print feat_imp[:30]
feat_imp[:30].plot(kind='barh', stacked=True, title='Feature Importances')
plt.savefig("xgb_fea_imp.jpg", format="jpg")

