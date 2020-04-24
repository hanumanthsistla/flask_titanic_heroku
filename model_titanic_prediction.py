#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:46:57 2020

@author: hduser
"""
import requests
import pandas
from sklearn.linear_model import LogisticRegression
import pickle
import json

# local url
url = 'http://localhost:8000' 

# Create sample data and convert to JSON:

# sample data

data = {'Pclass': 3
      , 'Age': 2
      , 'SibSp': 1
      , 'Fare': 50}

data = json.dumps(data)

# Post sample data and check response code using requests.post(url, data). 
# You want to get a response code of 200 to make sure that the app is working:

send_request = requests.post(url, data)
print(send_request)

# Then you can print the JSON of the request to see the model’s prediction:
    
print(send_request.json())
