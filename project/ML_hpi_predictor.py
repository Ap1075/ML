import pandas as pd
import quandl
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import numpy as np
from statistics import mean
from sklearn import svm, preprocessing, cross_validation
style.use('fivethirtyeight')

housing_data= pd.read_pickle('HPI.pickle')

housing_data= housing_data.pct_change() #pct_change gives the pct change from previous to next value

def create_labels(cur_hpi, fut_hpi):
    if fut_hpi>cur_hpi:
        return 1
    else:
        return 0
def moving_average(values):
    return mean(values)

housing_data.replace([np.inf, -np.inf], np.nan, inplace =True)
housing_data.dropna(inplace=True)

##print(housing_data.head())
housing_data['US_HPI_future']= housing_data['United States'].shift(-1)

housing_data.dropna(inplace=True)
##print(housing_data[['US_HPI_future','United States']].head())

housing_data['label']= list(map(create_labels,housing_data['United States'], housing_data['US_HPI_future']))
print(housing_data.head())

X = np.array(housing_data.drop(['label','US_HPI_future'],1))#features for ml. future not sure and label is going to be y hence removed
##print(X)
X= preprocessing.scale(X) #Scale X from -1 to 1
y= np.array(housing_data['label']) # labels to train and test on

X_train,X_test,y_train,y_test= cross_validation.train_test_split(X,y,test_size=0.2)

clf= svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
##print(clf.score(X_test,y_test))
print(housing_data[['United States','label']])
