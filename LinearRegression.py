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

from sklearn import linear_model
 ##Le modéle linéaire 
 
lm=linear_model.LinearRegression()
model=lm.fit(X,y)
