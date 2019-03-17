import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from subprocess import check_output

# fill NaNs with mean value
def fillna_mean(feature: pd.Series):
    mean = feature.mean()
    feature.fillna(mean, inplace=True)

# extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''

# print stats by feature
def print_feature(df:pd.DataFrame, label:str):
    ratio = df[label].value_counts() / df.shape[0]
    print( ratio.apply(lambda x: '{:.2f}'.format(x)).to_csv(header=True, sep='\t') )

# load data and setup features
def prepare_data(name: str) -> pd.Series:
    # load data
    df = pd.read_csv('../input/' + name + '.csv')
    print(name, df.shape)
    # fill in the missing values
    fillna_mean(df['Fare'])
    fillna_mean(df['Age'])
    fillna_mean(df['SibSp'])
    fillna_mean(df['Parch'])
    # new features - FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    print_feature(df, 'FamilySize')
    # new features - Alone
    df['Alone'] = 0
    df.loc[df['FamilySize'] == 1, 'Alone'] = 1
    print_feature(df, 'Alone')
    # new feature - Title
    df['Title'] = df['Name'].apply(get_title)
    df['Title'].replace('Mlle', 'Miss', inplace=True)
    df['Title'].replace('Ms', 'Miss', inplace=True)
    df['Title'].replace('Mme', 'Mrs', inplace=True)
    df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare', inplace=True)
    print_feature(df, 'Title')
    return df

def encode_labels(label: str):
    le = LabelEncoder()
    train[label].fillna('', inplace=True)
    test[label].fillna('', inplace=True)
    le.fit(train[label])
    train.loc[:, label] = le.transform(train[label])
    test.loc[:, label] = le.transform(test[label])

train = prepare_data('train')
test = prepare_data('test')

# Preprocess data
encode_labels('Sex')
encode_labels('Embarked')
encode_labels('Title')

# prepare train for fitting
# other features appears useless
features = ['Age', 'Sex', 'Pclass', 'Fare', 'Parch', 'Alone']
X = train.loc[:, features]
Y = train.loc[:, 'Survived']
XTest = test.loc[:, features]
XTrain, XValid, YTrain, YValid = train_test_split(X, Y, test_size=0.2, random_state=40)

# Fit logistic regression using scikit
LR = LogisticRegression(C=1000, solver='lbfgs', max_iter=1000)
LR.fit(X=XTrain, y=YTrain)

def accuracy(Y: np.array, yPred: np.array) -> float:
  return np.sum(yPred==Y) / len(Y)

# Use model to predict on training and validation sets
print('     Train accuracy', accuracy(YTrain, LR.predict(XTrain)))
print('Validation accuracy', accuracy(YValid, LR.predict(XValid)))

# Predict for test set
# Create a Kaggle submission
sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': LR.predict(XTest)})
sub.to_csv('scikitLRExample.csv', index=False)