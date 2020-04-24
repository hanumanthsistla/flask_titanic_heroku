#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:28:38 2020

@author: hduser
"""
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# create df
train = pd.read_csv('/home/hduser/Desktop/BF_Tuyeres_ML_Project/models_saved_dir/deploy-mlm-flask-heroku-master/titanic.csv') # change file path

# drop null values
train.dropna(inplace=True)

# features and target
target = 'Survived'
features = ['Pclass', 'Age', 'SibSp', 'Fare']
# X matrix, y vector
X = train[features]
y = train[target]
# model 
model = LogisticRegression()
model.fit(X, y)
model.score(X, y)
pickle.dump(model, open('/home/hduser/Desktop/BF_Tuyeres_ML_Project/models_saved_dir/deploy-mlm-flask-heroku-master/model_titanic.pkl', 'wb'))

print('\nSaved file: model_titanic.pkl to Directory: /home/hduser/Desktop/BF_Tuyeres_ML_Project/models_saved_dir/deploy-mlm-flask-heroku-master')