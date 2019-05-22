# Anton Panchenko 2019-05
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


class Smurf:
    def __init__(self, use_xgb, train, test):
        self.use_xgb = use_xgb
        # merge datasets
        self.data = train.append(test, sort=False)
        print('\nMeet data:\n', self.data.sample(5))
        # look at types and incomplete features
        print('\nTypes and counts:\n', self.count(self.data))

    def count(self, df):
        stat = pd.DataFrame([df.dtypes, df.count(), df.isna().sum()], index=['dtypes', 'values', 'nans'])
        return stat.sort_values(by=['values'], axis=1, ascending=False)

    def print_value_counts(self, msg, feature):
        print(msg)
        df = pd.DataFrame([feature.value_counts()], index=[feature.name])
        df['nan'] = feature.isna().sum()
        print(df)

    def encode_cat(self, label:str):
        notna = self.data[label].notna()
        y = self.data[notna].loc[:, label]
        self.data.loc[notna, label] = sl.preprocessing.LabelEncoder().fit_transform(y).astype('int32')
        self.print_value_counts('\nEncode categorical \'%s\':' % label, self.data[label])

    def infer(self, df, params, features:np.array, target:str):
        if self.use_xgb:
            self.infer_xgb(df, params, features, target)
        else:
            self.infer_linear(df, features, target)

    def infer_linear(self, df, features:np.array, target:str):
        new_feature = target.lower() + '_'
        print('find %s(%s):' % (new_feature, features))
        # select training data and fit regressor
        train = df[df[target].notna()]
        x = train.loc[:, features]
        y = train.loc[:, target]
        
        model = linear_model.LinearRegression()
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

    def infer_xgb(self, df, params, features:np.array, target:str):
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

    def infer_cat(self, df, params, features, target:str):
        if self.use_xgb:
            self.infer_cat_xgb(df, params, features, target)
        else:
            self.infer_cat_linear(df, params, features, target)

    def infer_cat_xgb(self, df, params, features, target:str):
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
        self.print_value_counts('', df[new_feature])

        # show feature importance
        feature_importance = pd.DataFrame({'feature':features, 'importance':estimator.feature_importances_})
        print('Feature importance:')
        print(feature_importance.sort_values(by='importance', ascending=False))

    def infer_cat_linear(self, df, params, features, target:str):
        new_feature = target.lower() + '_'
        print('\nInfer categorical %s(%s):' % (new_feature, features))  
        # select training data and classifier
        train = df[df[target].notna()]
        x = train.loc[:, features]
        y = train.loc[:, target]
        
        model = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
        model.fit(X=x, y=y)
        
        # predict missing target values 
        na = df[target].isna()
        x_predict = df[na].loc[:, features]
        y_predict = model.predict(x_predict)
        
        # create new feature
        df[new_feature] = df[target]
        df.loc[na, new_feature] = y_predict
        df[new_feature] = df[new_feature].astype('int64')
        self.print_value_counts('', df[new_feature])


class Titanic(Smurf):
    def __init__(self, use_xgb):
        self.train = pd.read_csv('../input/train.csv')
        self.test = pd.read_csv('../input/test.csv')
        Smurf.__init__(self, use_xgb, self.train, self.test)

    # Extract Title from Name
    def title(self):
        # See english honorifics (https://en.wikipedia.org/wiki/English_honorifics) for reference.
        self.data['title'] = self.data['Name'].str.extract(r', (.*?)\.', expand=False)
        self.print_value_counts('\nExtract title from name:\n', self.data['title'])
        self.data['title'].replace(['Mlle', 'Ms'], 'Miss', inplace=True)
        self.data['title'].replace(['Mme', 'Lady', 'Countess', 'Dona', 'the Countess'], 'Mrs', inplace=True)
        self.data['title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
        self.print_value_counts('\nReplace rare titles:\n', self.data['title'])
        self.encode_cat('title')

    # Encode Sex
    def sex(self):
        self.encode_cat('Sex')

    # Unite families
    def family(self):
        self.data['family'] = self.data['SibSp'] + self.data['Parch'] + 1
        self.print_value_counts('\nSibSp + Parch = family:', self.data['family'])

    # Check tickets
    def ticket(self):
        self.encode_cat('Ticket')

    # Pay Fare (best score 0.8489)
    def fare(self):
        #self.features = ['Pclass', 'title_cat', 'sex_cat', 'family', 'ticket_cat']
        self.features = ['Pclass', 'title_cat', 'sex_cat']
        params = {'max_depth': [2, 3, 4],
                'learning_rate': [0.3, 0.4, 0.5],
                'n_estimators': [150, 170, 190]}
        self.infer(self.data, params, self.features, 'Fare')

    # Encode and fix Embarked (best score 0.9479)
    def embarked(self):
        self.encode_cat('Embarked')
        params = {'max_depth': [3, 4, 5],
                'learning_rate': [0.4, 0.5, 0.6],
                'n_estimators': [400, 500, 600]}
        self.features = np.append(self.features, 'fare_')
        self.infer_cat(self.data, params, self.features, 'embarked_cat')

    # Fix Age (best score 0.4230)
    def age(self):
        self.features = np.append(self.features, 'embarked_cat_')
        params = {'max_depth': [2, 3],
                'learning_rate': [0.04, 0.05, 0.08],
                'n_estimators': [90, 100, 120]}
        self.infer(self.data, params, self.features, 'Age')        

    # Final glance at data
    def survived(self):
        # Finally predict Survived (best score 0.8350)
        features = np.append(self.features, 'age_')
        params = {'max_depth': [3, 4, 5],
                'learning_rate': [0.1, 0.3],
                'n_estimators': [70, 100, 300]}
        self.infer_cat(self.data, params, features, 'Survived')
        # create submission
        na_mask = self.data['Survived'].isna()
        sub = pd.DataFrame({'PassengerId': self.test['PassengerId'], 'Survived': self.data[na_mask].loc[:, 'survived_']})
        sub.to_csv('submission.csv', index=False)


if __name__=='__main__':
    titanic = Titanic(False)
    titanic.title()
    titanic.sex()
    titanic.family()
    titanic.ticket()
    titanic.fare()
    titanic.embarked()
    titanic.age()
    titanic.survived()