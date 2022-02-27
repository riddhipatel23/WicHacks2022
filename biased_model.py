import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('./data/adult.csv')
data.head()

#Data processing - missing values
data['occupation'].replace(to_replace ="?", 
                 value = "Miscellaneous", 
                  inplace = True)

# Label Encoding
le1=LabelEncoder()
occupation=le1.fit_transform(data['occupation'])
occ = dict(zip(occupation, data.occupation))

le2 = LabelEncoder()
gender = le2.fit_transform(data['gender'])
gen = dict(zip(gender, data.gender))

# print(occ)
# print(gen)

#updating the data with new labels
data.occupation = occupation
data.gender = gender

# defining parameters on which model will be trained
X = data[['age','occupation', 'gender']]
#desired output
Y = data['income']

#split data into training and testing model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 23)

#Training the model 
bias = OneVsRestClassifier(RandomForestClassifier(),n_jobs=-1)
bias.fit(X_train,Y_train)
# print(bias.predict([[40, 12, 1]]))
# print(bias.predict([[40, 12, 0]]))

#accuracy
bias.score(X_train,Y_train)











