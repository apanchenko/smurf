# 05/2019 Anton Panchenko
import sys
print(' python', sys.version)

import numpy as np
print('  numpy', np.__version__)

import pandas as pd
print(' pandas', pd.__version__)

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
import sklearn as sl
print('sklearn', sl.__version__)

import xgboost as xgb
print('xgboost', xgb.__version__)


def count(df):
    stat = pd.DataFrame([df.dtypes, df.count(), df.isna().sum()], index=['dtypes', 'values', 'nans'])
    return stat.sort_values(by=['values'], axis=1, ascending=False)

def value_counts(feature):
    df = pd.DataFrame([feature.value_counts()], index=[feature.name])
    df['nan'] = feature.isna().sum()
    return df

def encode_cat(df, label:str):
    target = label.lower() + '_cat'
    if target in df:
        df.drop(columns=target, inplace=True)
    notna = df[label].notna()
    y = df[notna].loc[:, label]
    df.loc[notna, target] = sl.preprocessing.LabelEncoder().fit_transform(y).astype('int32')
    print('\nEncode categorical \'%s\':' % label)
    print(value_counts(df[target]))

def infer_linear(df, features:np.array, target:str):
    new_feature = target.lower() + '_'
    print('find %s(%s):' % (new_feature, features))
    # select training data and fit regressor
    train = df[df[target].notna()]
    x = train.loc[:, features]
    y = train.loc[:, target]
    
    model = linear_model.LogisticRegression(C=1000, solver='lbfgs', max_iter=1000)
    model.fit(X=x, y=y)

    # predict missing target values
    na_mask = df[target].isna()
    predict = df[na_mask]
    x_predict = predict.loc[:, features]
    y_predict = model.predict(x_predict)

    # create new feature
    df[new_feature] = df[target]
    df.loc[na_mask, new_feature] = y_predict
    #df[new_feature].plot.kde()

def infer(df, params, features:np.array, target:str):
    new_feature = target.lower() + '_'
    print('\nInfer %s(%s):' % (new_feature, features))
    # select training data and fit regressor
    train = df[df[target].notna()]
    x = train.loc[:, features]
    y = train.loc[:, target]
    regressor = xgb.XGBRegressor(n_jobs=4)
    grid = sl.model_selection.GridSearchCV(regressor, params, cv=5, iid=True).fit(x, y)
    print('score', grid.best_score_)
    print('best params', grid.best_params_)
    regressor = grid.best_estimator_
    
    # predict missing target values
    na_mask = df[target].isna()
    predict = df[na_mask]
    x_predict = predict.loc[:, features]
    y_predict = regressor.predict(x_predict)

    # create new feature
    df[new_feature] = df[target]
    df.loc[na_mask, new_feature] = y_predict
    #df[new_feature].plot.kde()

    # show feature importance
    feature_importance = pd.DataFrame({'feature':features, 'importance':regressor.feature_importances_})
    print('Feature importance:')
    print(feature_importance.sort_values(by='importance', ascending=False))


def infer_cat(df, params, features, target:str):
    new_feature = target.lower() + '_'
    print('\nInfer categorical %s(%s):' % (new_feature, features))  
    # select training data and classifier
    train = df[df[target].notna()]
    x = train.loc[:, features]
    y = train.loc[:, target]
    estimator = xgb.XGBClassifier(n_jobs=4)
    grid = sl.model_selection.GridSearchCV(estimator, params, cv=3).fit(x, y)
    print('score', grid.best_score_)
    print('params', grid.best_params_)
    estimator = grid.best_estimator_
    
    # predict missing target values 
    na = df[target].isna()
    x_predict = df[na].loc[:, features]
    y_predict = estimator.predict(x_predict)
    
    # create new feature
    df[new_feature] = df[target]
    df.loc[na, new_feature] = y_predict
    df[new_feature] = df[new_feature].astype('int64')
    print(value_counts(df[new_feature]))

    # show feature importance
    feature_importance = pd.DataFrame({'feature':features, 'importance':estimator.feature_importances_})
    print('Feature importance:')
    print(feature_importance.sort_values(by='importance', ascending=False))

class Titanic:
    def __init__(self):
        # Load and merge datasets
        self.train = pd.read_csv('../input/train.csv')
        self.test = pd.read_csv('../input/test.csv')
        self.data = self.train.append(self.test, sort=False)
        print('\nMeet data:\n', self.data.sample(5))
        # Look at types and incomplete features
        print('\nTypes and counts:\n', count(self.data))

    # Extract Title from Name
    def title(self):
        # See english honorifics (https://en.wikipedia.org/wiki/English_honorifics) for reference.
        self.data['title'] = self.data['Name'].str.extract(r', (.*?)\.', expand=False)
        print('\nExtract title from name:\n', value_counts(self.data['title']))
        self.data['title'].replace(['Mlle', 'Ms'], 'Miss', inplace=True)
        self.data['title'].replace(['Mme', 'Lady', 'Countess', 'Dona', 'the Countess'], 'Mrs', inplace=True)
        self.data['title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
        print('\nReplace rare titles:\n', value_counts(self.data['title']))
        encode_cat(self.data, 'title')

    # Encode Sex
    def sex(self):
        encode_cat(self.data, 'Sex')

    # Unite families
    def family(self):
        self.data['family'] = self.data['SibSp'] + self.data['Parch'] + 1
        print('\nSibSp + Parch = family:')
        print(value_counts(self.data['family']))

    # Check tickets
    def ticket(self):
        encode_cat(self.data, 'Ticket')

    # Pay Fare (best score 0.8489)
    def fare(self):
        params = {'max_depth': [2, 3, 4],
                'learning_rate': [0.3, 0.4, 0.5],
                'n_estimators': [150, 170, 190]}
        self.features = ['Pclass', 'title_cat', 'sex_cat', 'family', 'ticket_cat']
        infer(self.data, params, self.features, 'Fare')
        #infer_linear(self.data, self.features, 'Fare')

    # Encode and fix Embarked (best score 0.9479)
    def embarked(self):
        encode_cat(self.data, 'Embarked')
        params = {'max_depth': [3, 4, 5],
                'learning_rate': [0.4, 0.5, 0.6],
                'n_estimators': [400, 500, 600]}
        self.features = np.append(self.features, 'fare_')
        infer_cat(self.data, params, self.features, 'embarked_cat')

    # Fix Age (best score 0.4230)
    def age(self):
        self.features = np.append(self.features, 'embarked_cat_')
        params = {'max_depth': [2, 3],
                'learning_rate': [0.04, 0.05, 0.08],
                'n_estimators': [90, 100, 120]}
        infer(self.data, params, self.features, 'Age')        

    # Final glance at data
    def survived(self):
        print('\nFinal data:\n', count(self.data))
        # Finally predict Survived (best score 0.8350)
        features = np.append(self.features, 'age_')
        params = {'max_depth': [3, 4, 5],
                'learning_rate': [0.1, 0.3],
                'n_estimators': [70, 100, 300]}
        infer_cat(self.data, params, features, 'Survived')
        # create submission
        na_mask = self.data['Survived'].isna()
        sub = pd.DataFrame({'PassengerId': self.test['PassengerId'], 'Survived': self.data[na_mask].loc[:, 'survived_']})
        sub.to_csv('submission.csv', index=False)


if __name__=='__main__':
    titanic = Titanic()
    titanic.title()
    titanic.sex()
    titanic.family()
    titanic.ticket()
    titanic.fare()
    titanic.embarked()
    titanic.age()
    titanic.survived()