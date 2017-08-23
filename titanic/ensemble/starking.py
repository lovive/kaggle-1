import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score

def submit(pre):
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pre
    })
    print "write submit file:decisiontree.csv"
    submission.to_csv('../data/stacking.csv', encoding="utf-8", index=False)


if __name__ == '__main__':
    train = pd.read_csv('../data/feature_train.csv')
    test = pd.read_csv('../data/feature_test.csv')

    features = ["Pclass", "Fare", "Age", "sex", "fimalysize",
                "embark", "name", 'cabin']
    x_train = train[features]
    y_train = train["Survived"]
    x_test = test[features]

    clfs = [DecisionTreeClassifier(),
            RandomForestClassifier(n_estimators=140, max_depth=4, min_samples_split=6, min_samples_leaf=4, n_jobs=4),
            GradientBoostingClassifier(n_estimators=140, max_depth=4, min_samples_split=6, min_samples_leaf=4)]

    ntrain = train.shape[0]
    ntest = test.shape[0]
    kf = KFold(ntrain, n_folds=5, random_state=0)
    print "Creating train and test sets for blending."

    def get_oof(clf, x_train, y_train, x_test):
        S_test_i = test.shape[0]
        oof_train = np.zeros((ntrain, ))
        oof_test = np.zeros((ntest, ))
        oof_test_skf = np.empty((5, ntest))
        for i, (train_index, test_index) in enumerate(kf):
            print "Kflod", i+1
            kf_x_train = x_train[train_index]
            kf_y_train = y_train[train_index]
            kf_x_test = x_train[test_index]

            clf.fit(kf_x_train, kf_y_train)

            oof_train[test_index] = clf.predict(kf_x_test)
            oof_test_skf[i, :] = clf.predict(x_test)
        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    rf = RandomForestClassifier(n_estimators=140, max_depth=4, min_samples_split=6, min_samples_leaf=4, n_jobs=4)
    dt = DecisionTreeClassifier()
    gb = GradientBoostingClassifier(n_estimators=140, max_depth=4, min_samples_split=6, min_samples_leaf=4)

    rf_oof_train, rf_oof_test = get_oof(rf, x_train.values, y_train, test[features].values)
    dt_oof_train, dt_oof_test = get_oof(dt, x_train.values, y_train, test[features].values)
    gb_oof_train, gb_oof_test = get_oof(gb, x_train.values, y_train, test[features].values)

    print "Train is complete"

    base_prediction_train = pd.DataFrame({
        "randomforest": rf_oof_train.ravel(),
        "decisiontree": dt_oof_train.ravel(),
        "gradientboost": gb_oof_train.ravel(),
    })
    print base_prediction_train.head(3)

    xgb_train = np.concatenate((rf_oof_train, dt_oof_train, gb_oof_train), axis=1)
    xgb_test = np.concatenate((rf_oof_test, dt_oof_test, gb_oof_test), axis=1)

    gbm = xgb.XGBClassifier(
        learning_rate=0.005,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=2,
        #gamma=1,
        gamma=0.9,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1)
    scores = cross_val_score(gbm, xgb_train, y_train, cv=5)
    print scores
    gbm.fit(xgb_train, y_train)
    predictions = gbm.predict(xgb_test)
    submit(predictions)




