# Anton Panchenko
import xgboost as xgb
from sklearn import model_selection
import pandas as pd
import numpy as np
from classifier import Classifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Regressor(Classifier):

    def infer(self, params, features: np.array, label: str):
        print('Infer ', label)

        # select training data
        train = self.data[self.data[label].notna()]
        x = train.loc[:, features]
        y = train.loc[:, label]

        # search params
        regressor = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=4)
        model = model_selection.GridSearchCV(regressor, params, cv=10, refit=True)
        model.fit(x, y)
        print('Best params', model.best_params_)
        feature_importance = pd.DataFrame({'feature': features, 'importance': model.best_estimator_.feature_importances_})
        print('Feature importance:')
        print(feature_importance.sort_values(by='importance', ascending=False))

        # predict missing target values
        na = self.data[label].isna()
        predict = self.data[na]
        x_predict = predict.loc[:, features]
        self.data.loc[na, label] = model.predict(x_predict)