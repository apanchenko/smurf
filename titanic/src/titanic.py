from smurf import Smurf
import pandas as pd
import numpy as np

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
