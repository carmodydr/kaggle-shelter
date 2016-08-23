
import numpy as np

import pandas as pd

import seaborn as sns


# In[2]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import cross_validation


train_df = pd.read_csv('train.csv')



def ageCalc(s):
    if s is None: return -1
    
    ageStr = str(s).split(' ')
    age = -1
    try:
        if 'year' in ageStr[1]:
            age = int(ageStr[0]) * 365
        elif 'day' in ageStr[1]:
            age = int(ageStr[0])
        elif 'month' in ageStr[1]:
            age = int(ageStr[0]) * 30
        elif 'week' in ageStr[1]:
            age = int(ageStr[0]) * 7
    except:
        pass
    
    return age

def ageSimple(s):
    if s is None: return -1
    
    try:
        ageStr = str(s).split(' ')
        age = -1
        if 'year' in ageStr[1]:
            age = 3
        elif 'day' in ageStr[1]:
            age = 0
        elif 'month' in ageStr[1]:
            age = 2
        elif 'week' in ageStr[1]:
            age = 1
    except:
        pass
            
    return age

def timeOfDay(s):
    if s is None: return -1
    
    time = pd.to_datetime(s)
    ToD = -1
    if time.hour < 5:
        ToD = 0
    elif time.hour < 11:
        ToD = 1
    elif time.hour < 16:
        ToD = 2
    elif time.hour < 20:
        ToD = 3
    
    return ToD



def prepareData(df, train_flag):
    returnDF = pd.DataFrame()
    if train_flag:
        returnDF['OType'] = df.OutcomeType.map( {'Return_to_owner': 0, 'Euthanasia': 1, 'Transfer': 2, 'Adoption': 3, 'Died': 4})
        returnDF['OutcomeType'] = df.OutcomeType
    returnDF['AType'] = df.AnimalType.map( {'Dog': 0, 'Cat': 1})
    returnDF['Sex'] = df.SexuponOutcome.map( {'Neutered Male': 0, 'Spayed Female': 1, 'Intact Male': 2, 'Intact Female': 3, 'Unknown': -1})
    returnDF['Intact'] = df.SexuponOutcome.map( {'Neutered Male': 0, 'Spayed Female': 0, 'Intact Male': 1, 'Intact Female': 1, 'Unknown': -1})
    returnDF['NoName'] = df.Name.isnull().astype(int)
    returnDF['DoW'] = pd.to_datetime(df.DateTime).dt.day
    returnDF['Month'] = pd.to_datetime(df.DateTime).dt.month
    returnDF['AgeDays'] = df.AgeuponOutcome.map(ageCalc)
    returnDF['AgeSimple'] = df.AgeuponOutcome.map(ageSimple)
    returnDF['Time'] = df.DateTime.map(timeOfDay)
    returnDF['Breed'] = df.Breed.map({
            'Domestic Shorthair Mix': 0,
            'Pit Bull Mix': 1,
    'Chihuahua Shorthair Mix': 2,
    'Labrador Retriever Mix': 3,
    'Domestic Medium Hair Mix': 4,
    'German Shepherd Mix': 5,
    'Domestic Longhair Mix': 6,
    'Siamese Mix': 7,
    'Australian Cattle Dog Mix': 8,
    'Dachshund Mix': 9,
    'Boxer Mix': 10,
    'Miniature Poodle Mix': 11,
    'Border Collie Mix': 12
        })
    
    returnDF.Breed = returnDF.Breed.fillna(-1)
    
    returnDF.dropna(inplace=True)
    
    return returnDF


# In[7]:

prepTraindf = prepareData(train_df, 1)

def shelterScore(prediction,actual):
    n = len(prediction)
    m = 5
    logloss = 0
    for i in range(n):
        for j in range(m):
            logloss += -(1.0/n)*actual[i,j]*np.log(prediction[i,j])
    return logloss


outcomeTypes = pd.get_dummies(prepTraindf.OutcomeType)


# In[32]:

features = ["AType","AgeSimple","Intact","Time","Breed","NoName"]
x_train, x_test, y_train, y_test = cross_validation.train_test_split(prepTraindf[features],outcomeTypes, test_size=0.3)


# # Logisitic Regression

# In[33]:

x_train_lr, x_test_lr, y_train_lr, y_test_lr = cross_validation.train_test_split(prepTraindf[features],prepTraindf.OType, test_size=0.3)


# In[ ]:

LR = LogisticRegression(penalty='l2',C=1.0)
LR.fit(x_train_lr,y_train_lr)


# # Random Forest Classifier

# In[34]:

def run_fit(ne, md):
    forest = RandomForestClassifier(n_estimators=ne, max_depth=md)
    forest.fit(x_train,y_train)
    outScore = forest.predict_proba(x_test)
    
    scoreL = np.zeros((len(outScore[0]),len(outScore)))
    i = 0
    for cl in outScore:
        j = 0
        for animal in cl:
            scoreL[j][i] = animal[1]
            j += 1
        i += 1
    
    return shelterScore(scoreL,np.array(y_test))

#run_fit(10,4)





