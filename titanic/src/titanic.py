# 04/2019 Anton Panchenko
from sys import version
import re
import numpy as np
import pandas as pd
#import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
import xgboost as xgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def count(df):
    stat = pd.DataFrame([df.dtypes, df.count(), df.isna().sum()], index=['dtypes', 'values', 'nans'])
    return stat.sort_values(by=['values'], axis=1, ascending=False)

def print_value_counts(feature):
    df = pd.DataFrame([feature.value_counts()], index=[feature.name])
    df['nan'] = feature.isna().sum()
    print(df)

def encode_cat(df, label:str):
    target = label.lower() + '_cat'
    if target in df:
        df.drop(columns=target, inplace=True)
    notna = df[label].notna()
    y = df[notna].loc[:, label]
    df.loc[notna, target] = LabelEncoder().fit_transform(y).astype('int32')
    print('\nencode_cat ', label)
    print_value_counts(df[target])

def infer(df, params, features:np.array, target:str):
    # select training data and fit regressor
    train = df[df[target].notna()]
    x = train.loc[:, features]
    y = train.loc[:, target]
    regressor = xgb.XGBRegressor(n_jobs=4)
    grid = model_selection.GridSearchCV(regressor, params, cv=5).fit(x, y)
    print('score', grid.best_score_)
    print('params', grid.best_params_)
    regressor = grid.best_estimator_
    
    # predict missing target values
    na_mask = df[target].isna()
    predict = df[na_mask]
    x_predict = predict.loc[:, features]
    y_predict = regressor.predict(x_predict)

    # create new feature
    new_feature = target + '_'
    df[new_feature] = df[target]
    df.loc[na_mask, new_feature] = y_predict
    #df[new_feature].plot.kde()

    # return feature importance
    #feature_importance = pd.DataFrame({'feature':features, 'importance':regressor.feature_importances_})
    #return feature_importance.sort_values(by='importance', ascending=False)

def infer_cat(df, params, features, target:str):
    # select training data and classifier
    train = df[df[target].notna()]
    x = train.loc[:, features]
    y = train.loc[:, target]
    estimator = xgb.XGBClassifier(n_jobs=4)
    grid = model_selection.GridSearchCV(estimator, params, cv=3).fit(x, y)
    print('score', grid.best_score_)
    print('params', grid.best_params_)
    estimator = grid.best_estimator_
    
    # predict missing target values 
    na = df[target].isna()
    x_predict = df[na].loc[:, features]
    y_predict = estimator.predict(x_predict)
    
    # create new feature
    new_feature = target + '_'
    df[new_feature] = df[target]
    df.loc[na, new_feature] = y_predict
    df[new_feature] = df[new_feature].astype('int64')
    print_value_counts(df[new_feature])


def main():
  # Load and merge datasets
  train = pd.read_csv('../input/train.csv')
  test = pd.read_csv('../input/test.csv')
  data = train.append(test, sort=False)
  print('\nMeet data:\n', data.sample(5))

  # Look at types and incomplete features
  print('\nTypes and counts:\n', count(data))
  # Features to make categorical:
  #    - Name
  #    - Sex
  #    - Ticket
  #    - Embarked
  # Incomplete features are:
  #    - Fare
  #    - Embarked
  #    - Age
  #    - Cabin

  # Extract Title from Name
  # See english honorifics (https://en.wikipedia.org/wiki/English_honorifics) for reference.
  data['title'] = data['Name'].str.extract(r', (.*?)\.', expand=False)
  print_value_counts(data['title'])
  data['title'].replace(['Mlle', 'Ms'], 'Miss', inplace=True)
  data['title'].replace(['Mme', 'Lady', 'Countess', 'Dona', 'the Countess'], 'Mrs', inplace=True)
  data['title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
  print_value_counts(data['title'])
  encode_cat(data, 'title')

  # Check tickets
  encode_cat(data, 'Ticket')

  # Create FamilySize and Alone features
  data['family_size'] = data['SibSp'] + data['Parch'] + 1
  data['alone'] = 0
  data.loc[data['family_size'] == 1, 'alone'] = 1
  print_value_counts(data['alone'])

  # Encode Sex
  encode_cat(data, 'Sex')

  ## Fix Fare
  params = {'max_depth': [2, 3, 4],
            'learning_rate': [0.3, 0.4, 0.5],
            'n_estimators': [150, 170, 190]}
  fare_features = ['Pclass', 'SibSp', 'sex_cat', 'title_cat', 'ticket_cat', 'family_size']
  infer(data, params, fare_features, 'Fare')

  # ## Encode and fix Embarked
  encode_cat(data, 'Embarked')
  params = {'max_depth': [3, 4, 5],
            'learning_rate': [0.4, 0.5, 0.6],
            'n_estimators': [400, 500, 600]}
  emb_features = np.append(fare_features, 'Fare_')
  infer_cat(data, params, emb_features, 'embarked_cat')

  # ## Fix Age
  infer(data, params, ['Pclass', 'SibSp', 'sex_cat', 'title_cat', 'ticket_cat', 'family_size', 'Fare_'], 'Age')

  # ## Finally predict Survived
  infer_cat(data, params, ['Pclass', 'SibSp', 'Parch', 'title_cat', 'sex_cat', 'Fare_', 'Age_', 'embarked_cat'], 'Survived')
  na_mask = data['Survived'].isna()

  # create a Kaggle submission
  sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': data[na_mask].loc[:, 'Survived_']})
  sub.to_csv('submission.csv', index=False)


if __name__=='__main__':
  main()