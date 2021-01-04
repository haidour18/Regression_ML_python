#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 14:21:31 2021

@author: Mac
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import le dataset
dataset= pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values
#Encoding Categorical data les X
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
Labelencoder_X=LabelEncoder()
X[:,3]=Labelencoder_X.fit_transform(X[:,3])
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)

#Spliting the dataset into training dataset and test dataset
X=X[:,1:]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#You don't need to sacle your data the library take care 

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
error= regressor.score(X_train, y_train)
print('coefficient of determination:',error)
print('intercept:a', regressor.intercept_)
print('slope:b', regressor.coef_)

y_pred = regressor.predict(X_test)

#Building a model using backward elimination 
import statsmodels.api as sm
#add b1X1 to oour model with x1=1 Step 1
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
#Step  fit the full model with all possible predictors
X_opt = X[:,[0,1,2,3,4,5]]
X_opt = np.array(X_opt, dtype=float) 
regressor_OLS = sm.OLS(endog =y ,exog =X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:,[0,1,3,4,5]]
X_opt = np.array(X_opt, dtype=float) 
regressor_OLS = sm.OLS(endog =y ,exog =X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:,[0,3,4,5]]
X_opt = np.array(X_opt, dtype=float) 
regressor_OLS = sm.OLS(endog =y ,exog =X_opt).fit()
print(regressor_OLS.summary())


X_opt = X[:,[0,3,5]]
X_opt = np.array(X_opt, dtype=float) 
regressor_OLS = sm.OLS(endog =y ,exog =X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:,[0,3]]
X_opt = np.array(X_opt, dtype=float) 
regressor_OLS = sm.OLS(endog =y ,exog =X_opt).fit()
print(regressor_OLS.summary())
