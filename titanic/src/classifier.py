# Anton Panchenko 2019-05
import xgboost as xgb
import sklearn as sl
from sklearn import linear_model
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print(' python', sys.version)

print('  numpy', np.__version__)

print(' pandas', pd.__version__)

print('sklearn', sl.__version__)

print('xgboost', xgb.__version__)


class Classifier:
    def __init__(self, n_jobs, train, test):
        print('===='*20)
        self.n_jobs = n_jobs
        # merge datasets
        self.data = train.append(test, sort=False)
        print('Meet data:')
        print(self.data.sample(5))
        # look at types and incomplete features
        print('\nTypes and counts:')
        print(self.count())

    def print_cutline(self):
        print('\n')
        print('----'*20)

    def count(self):
        stat = pd.DataFrame(
            [self.data.dtypes, self.data.count(), self.data.isna().sum()],
            index=['dtypes', 'values', 'nans'])
        return stat.sort_values(by=['values'], axis=1, ascending=False)

    def value_counts(self, label : str):
        feature = self.data[label]
        df = pd.DataFrame([feature.value_counts()], index=[feature.name + ' ' + str(feature.dtypes)])
        df['nan'] = feature.isna().sum()
        return df

    def encode_cat(self, label: str):
        target = label.lower() + '_cat'
        notna = self.data[label].notna()
        y = self.data[notna].loc[:, label]
        self.data.loc[notna, target] = LabelEncoder().fit_transform(y).astype('int32')
        self.data[label] = self.data[target]
        print('Encode categorical \'%s\':' % label)
        print(self.value_counts(label))

    def infer_cat(self, params, features, label: str):
        print('Infer categorical', label)
        # select training data and classifier
        train = self.data[self.data[label].notna()]
        x = train.loc[:, features]
        y = train.loc[:, label]
        xt, xv, yt, yv = train_test_split(x, y, test_size=0.2, random_state=40)

        model = self._infer_cat_linear(xt, yt)
        score = model.score(xv, yv)
        print('Linear score {:.4f}'.format(score))

        xgb_model = self._infer_cat_xgb(params, xt, yt, features)
        xgb_score = xgb_model.score(xv, yv)
        print('XGB score {:.4f}'.format(xgb_score))
        if xgb_score > score:
            model = xgb_model

        # predict missing target values
        na = self.data[label].isna()
        test = self.data[na]
        self.data.loc[na, label] = model.predict(test.loc[:, features])
        self.data[label] = self.data[label].astype('int32')
        print(self.value_counts(label))        

    def _infer_cat_xgb(self, params, x, y, features):
        model = xgb.XGBClassifier(n_jobs=self.n_jobs)
        grid = sl.model_selection.GridSearchCV(model, params, cv=3).fit(x, y)
        print('best params', grid.best_params_)
        model = grid.best_estimator_
        feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        print('Feature importance:')
        print(feature_importance.sort_values(by='importance', ascending=False))
        return model

    def _infer_cat_linear(self, x, y):
        model = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)
        model.fit(x, y)
        return model
