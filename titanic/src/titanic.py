# Anton Panchenko 2019-05
import xgboost as xgb
import sklearn as sl
from sklearn import linear_model
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import sys
print(' python', sys.version)

print('  numpy', np.__version__)

print(' pandas', pd.__version__)

print('sklearn', sl.__version__)

print('xgboost', xgb.__version__)


class Smurf:
    def __init__(self, use_xgb, train, test):
        self.use_xgb = use_xgb
        # merge datasets
        self.data = train.append(test, sort=False)
        print('\nMeet data:\n', self.data.sample(5))
        # look at types and incomplete features
        print('\nTypes and counts:\n', self.count())

    def count(self):
        stat = pd.DataFrame(
            [self.data.dtypes, self.data.count(), self.data.isna().sum()],
            index=['dtypes', 'values', 'nans'])
        return stat.sort_values(by=['values'], axis=1, ascending=False)

    def print_value_counts(self, msg, feature):
        print(msg)
        df = pd.DataFrame([feature.value_counts()], index=[feature.name])
        df['nan'] = feature.isna().sum()
        print(df)

    def encode_cat(self, label: str):
        target = label.lower() + '_cat'
        notna = self.data[label].notna()
        y = self.data[notna].loc[:, label]
        self.data.loc[notna, target] = LabelEncoder().fit_transform(y).astype('int32')
        self.print_value_counts('\nEncode categorical \'%s\':' % label, self.data[target])

    def infer(self, params, features: np.array, target: str):
        if self.use_xgb:
            self.infer_xgb(params, features, target)
        else:
            self.infer_linear(features, target)

    def infer_linear(self, features: np.array, label: str):
        print('find %s(%s):' % (label, features))
        # select training data and fit regressor
        train = self.data[self.data[label].notna()]
        x = train.loc[:, features]
        y = train.loc[:, label]

        model = linear_model.LinearRegression()
        model.fit(x, y)

        # predict missing target values
        na_mask = self.data[label].isna()
        predict = self.data[na_mask]
        x_predict = predict.loc[:, features]
        self.data.loc[na_mask, label] = model.predict(x_predict)

    def infer_xgb(self, params, features: np.array, label: str):
        print('\nInfer %s(%s):' % (label, features))
        # select training data and fit regressor
        train = self.data[self.data[label].notna()]
        x = train.loc[:, features]
        y = train.loc[:, label]
        regressor = xgb.XGBRegressor(n_jobs=4)
        grid = sl.model_selection.GridSearchCV(regressor, params, cv=5, iid=True).fit(x, y)
        print('score', grid.best_score_)
        print('best params', grid.best_params_)
        regressor = grid.best_estimator_

        # predict missing target values
        na = self.data[label].isna()
        predict = self.data[na]
        x_predict = predict.loc[:, features]
        self.data.loc[na, label] = regressor.predict(x_predict)

        # show feature importance
        feature_importance = pd.DataFrame({'feature': features, 'importance': regressor.feature_importances_})
        print('Feature importance:')
        print(feature_importance.sort_values(by='importance', ascending=False))

    def infer_cat(self, params, features, target: str):
        if self.use_xgb:
            self.infer_cat_xgb(params, features, target)
        else:
            self.infer_cat_linear(params, features, target)

    def infer_cat_xgb(self, params, features, target: str):
        new_feature = target.lower() + '_'
        print('\nInfer categorical %s(%s):' % (new_feature, features))
        # select training data and classifier
        train = self.data[self.data[target].notna()]
        x = train.loc[:, features]
        y = train.loc[:, target]
        estimator = xgb.XGBClassifier(n_jobs=4)
        grid = sl.model_selection.GridSearchCV(estimator, params, cv=3).fit(x, y)
        print('score', grid.best_score_)
        print('params', grid.best_params_)
        estimator = grid.best_estimator_

        # predict missing target values
        na = self.data[target].isna()
        x_predict = self.data[na].loc[:, features]
        y_predict = estimator.predict(x_predict)

        # create new feature
        self.data[new_feature] = self.data[target]
        self.data.loc[na, new_feature] = y_predict
        self.data[new_feature] = self.data[new_feature].astype('int64')
        self.print_value_counts('', self.data[new_feature])

        # show feature importance
        feature_importance = pd.DataFrame(
            {'feature': features, 'importance': estimator.feature_importances_})
        print('Feature importance:')
        print(feature_importance.sort_values(by='importance', ascending=False))

    def infer_cat_linear(self, params, features, target: str):
        new_feature = target.lower() + '_'
        print('\nInfer categorical %s(%s):' % (new_feature, features))
        # select training data and classifier
        train = self.data[self.data[target].notna()]
        x = train.loc[:, features]
        y = train.loc[:, target]

        model = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)
        model.fit(X=x, y=y)

        # predict missing target values
        na = self.data[target].isna()
        x_predict = self.data[na].loc[:, features]
        y_predict = model.predict(x_predict)

        # create new feature
        self.data[new_feature] = self.data[target]
        self.data.loc[na, new_feature] = y_predict
        self.data[new_feature] = self.data[new_feature].astype('int64')
        self.print_value_counts('', self.data[new_feature])


class Titanic(Smurf):
    def __init__(self, use_xgb):
        self.train = pd.read_csv('../input/train.csv')
        self.test = pd.read_csv('../input/test.csv')
        Smurf.__init__(self, use_xgb, self.train, self.test)

    # Extract Title from Name
    def title(self):
        # See english honorifics (https://en.wikipedia.org/wiki/English_honorifics) for reference.
        self.data['title'] = self.data['Name'].str.extract(r', (.*?)\.', expand=False)
        self.print_value_counts('\nExtract title from name:', self.data['title'])
        self.data['title'].replace(['Mlle', 'Ms'], 'Miss', inplace=True)
        self.data['title'].replace(['Mme', 'Lady', 'Countess', 'Dona', 'the Countess'], 'Mrs', inplace=True)
        self.data['title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
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
        self.features = ['Pclass', 'title_cat', 'sex_cat', 'family', 'ticket_cat']
        #self.features = ['Pclass', 'title_cat', 'sex_cat']
        params = {'max_depth': [2, 3, 4],
                  'learning_rate': [0.3, 0.4, 0.5],
                  'n_estimators': [150, 170, 190]}
        self.infer(params, self.features, 'Fare')

    # Encode and fix Embarked (best score 0.9479)
    def embarked(self):
        self.encode_cat('Embarked')
        params = {'max_depth': [3, 4, 5],
                  'learning_rate': [0.4, 0.5, 0.6],
                  'n_estimators': [400, 500, 600]}
        self.features = np.append(self.features, 'Fare')
        self.infer_cat(params, self.features, 'embarked_cat')

    # Fix Age (best score 0.4230)
    def age(self):
        self.features = np.append(self.features, 'embarked_cat_')
        params = {'max_depth': [2, 3],
                  'learning_rate': [0.04, 0.05, 0.08],
                  'n_estimators': [90, 100, 120]}
        self.infer(params, self.features, 'Age')

    # Final glance at data
    def survived(self):
        # Finally predict Survived (best score 0.8350)
        features = np.append(self.features, 'Age')
        params = {'max_depth': [3, 4, 5],
                  'learning_rate': [0.1, 0.3],
                  'n_estimators': [70, 100, 300]}
        self.infer_cat(params, features, 'Survived')
        # create submission
        na_mask = self.data['Survived'].isna()
        sub = pd.DataFrame(
            {'PassengerId': self.test['PassengerId'], 'Survived': self.data[na_mask].loc[:, 'survived_']})
        sub.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    titanic = Titanic(False)
    titanic.title()
    titanic.sex()
    titanic.family()
    titanic.ticket()
    titanic.fare()
    titanic.embarked()
    titanic.age()
    titanic.survived()
