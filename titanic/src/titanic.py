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
print(' python', sys.version)

print('  numpy', np.__version__)

print(' pandas', pd.__version__)

print('sklearn', sl.__version__)

print('xgboost', xgb.__version__)


class Smurf:
    def __init__(self, use_xgb, n_jobs, train, test):
        print('===='*20)
        self.use_xgb = use_xgb
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

    def accuracy(self, y, yPred) -> float:
        return np.sum(yPred == y) / len(y)

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

    def infer(self, params, features: np.array, label: str):
        print('Infer ', label)
        # select training data
        train = self.data[self.data[label].notna()]
        x = train.loc[:, features]
        y = train.loc[:, label]
        xt, xv, yt, yv = train_test_split(x, y, test_size=0.2, random_state=40)

        model = None
        if self.use_xgb:
            model = self._infer_xgb(params, xt, xv, yt, yv)
        else:
            model = self._infer_linear(xt, xv, yt, yv)

        # predict missing target values
        na = self.data[label].isna()
        predict = self.data[na]
        x_predict = predict.loc[:, features]
        self.data.loc[na, label] = model.predict(x_predict)

    def _infer_linear(self, xt, xv, yt, yv):
        model = linear_model.LinearRegression(normalize=True, n_jobs=self.n_jobs)
        model.fit(xt, yt)
        print('Score {:.4f}'.format(model.score(xv, yv)))
        return model

    def _infer_xgb(self, params, xt, xv, yt, yv):
        regressor = xgb.XGBRegressor(n_jobs=4)
        grid = sl.model_selection.GridSearchCV(regressor, params, cv=5, iid=True).fit(xt, yt)
        print('score', grid.best_score_)
        print('best params', grid.best_params_)
        model = grid.best_estimator_
        # show feature importance
        #feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        #print('Feature importance:')
        #print(feature_importance.sort_values(by='importance', ascending=False))
        return model

    def infer_cat(self, params, features, label: str):
        print('Infer categorical', label)
        # select training data and classifier
        train = self.data[self.data[label].notna()]
        x = train.loc[:, features]
        y = train.loc[:, label]
        xt, xv, yt, yv = train_test_split(x, y, test_size=0.2, random_state=40)

        model = None
        if self.use_xgb:
            model = self._infer_cat_xgb(params, xt, xv, yt, yv)
        else:
            model = self._infer_cat_linear(xt, xv, yt, yv)

        # predict missing target values
        na = self.data[label].isna()
        test = self.data[na]
        self.data.loc[na, label] = model.predict(test.loc[:, features])
        self.data[label] = self.data[label].astype('int32')
        print(self.value_counts(label))        

    def _infer_cat_xgb(self, params, xt, xv, yt, yv):
        model = xgb.XGBClassifier(n_jobs=self.n_jobs)
        grid = sl.model_selection.GridSearchCV(model, params, cv=3).fit(xt, yt)
        print('score', grid.best_score_)
        print('params', grid.best_params_)
        model = grid.best_estimator_

        # show feature importance
        #feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        #print('Feature importance:')
        #print(feature_importance.sort_values(by='importance', ascending=False))
        return model

    def _infer_cat_linear(self, xt, xv, yt, yv):
        model = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)
        model.fit(xt, yt)
        print('Accuracy', model.score(xv, yv))
        return model


class Titanic(Smurf):
    def __init__(self, use_xgb, n_jobs):
        self.train = pd.read_csv('../input/train.csv')
        self.test = pd.read_csv('../input/test.csv')
        Smurf.__init__(self, use_xgb, n_jobs, self.train, self.test)

    # Extract Title from Name
    def title(self):
        self.print_cutline()
        # See english honorifics (https://en.wikipedia.org/wiki/English_honorifics) for reference.
        self.data['Title'] = self.data['Name'].str.extract(r', (.*?)\.', expand=False)
        print('Extract Title from Name:')
        print(self.value_counts('Title'), '\n')
        self.data['Title'].replace(['Mlle', 'Ms'], 'Miss', inplace=True)
        self.data['Title'].replace(['Mme', 'Lady', 'Countess', 'Dona', 'the Countess'], 'Mrs', inplace=True)
        self.data['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
        self.encode_cat('Title')

    # Encode Sex
    def sex(self):
        self.print_cutline()
        self.encode_cat('Sex')

    # Unite families
    def family(self):
        self.print_cutline()
        self.data['Family'] = self.data['SibSp'] + self.data['Parch'] + 1
        print('Family = SibSp + Parch + 1')
        print(self.value_counts('Family'))

    # Check tickets
    def ticket(self):
        self.print_cutline()
        self.encode_cat('Ticket')

    # Pay Fare (best score 0.8489)
    def fare(self):
        self.print_cutline()
        self.features = ['Pclass', 'Title', 'Sex', 'Family', 'Ticket']
        params = {'max_depth': [2, 3, 4],
                  'learning_rate': [0.3, 0.4, 0.5],
                  'n_estimators': [150, 170, 190]}
        self.infer(params, self.features, 'Fare')
        self.features = np.append(self.features, 'Fare')

    # Encode and fix Embarked (best score 0.9479)
    def embarked(self):
        self.print_cutline()
        self.encode_cat('Embarked')
        params = {'max_depth': [3, 4, 5],
                  'learning_rate': [0.4, 0.5, 0.6],
                  'n_estimators': [400, 500, 600]}
        self.infer_cat(params, self.features, 'Embarked')
        self.features = np.append(self.features, 'Embarked')

    # Fix Age (best score 0.4230)
    def age(self):
        self.print_cutline()
        params = {'max_depth': [2, 3],
                  'learning_rate': [0.04, 0.05, 0.08],
                  'n_estimators': [90, 100, 120]}
        self.infer(params, self.features, 'Age')
        self.features = np.append(self.features, 'Age')

    # Final glance at data
    def survived(self):
        self.print_cutline()
        # Finally predict Survived (best score 0.8350)
        params = {'max_depth': [3, 4, 5],
                  'learning_rate': [0.1, 0.3],
                  'n_estimators': [70, 100, 300]}
        self.infer_cat(params, self.features, 'Survived')
        # create submission
        na = self.data['Survived'].isna()
        sub = pd.DataFrame({'PassengerId': self.test['PassengerId'], 'Survived': self.data[na].loc[:, 'Survived']})
        sub.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    titanic = Titanic(use_xgb=False, n_jobs=4)
    titanic.title()
    titanic.sex()
    titanic.family()
    titanic.ticket()
    titanic.fare()
    titanic.embarked()
    titanic.age()
    titanic.survived()
