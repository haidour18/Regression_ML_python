#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 22:16:06 2021

@author: Mac
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importer le dataset
dataset= pd.read_csv('Salary_Data.csv')
#Extraire les variables indépendantes 
X=dataset.iloc[:,:-1].values
#Extraire la variables dépendante
y=dataset.iloc[:,1].values
#Afficher  le graphe 
dataset.plot(kind='scatter',x="YearsExperience",y="Salary")
#Corrélation 
dataset.corr()
# Splitting the dataset into the Training set and Test set
from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
#Le modéle linéaire 
 
model=LinearRegression()
model.fit(X,y)
error= model.score(X_train, y_train)
print('coefficient of determination:',error)
print('intercept:', model.intercept_)
print('slope:', model.coef_)



# Predicting the Test set results
y_pred = model.predict(X_test)