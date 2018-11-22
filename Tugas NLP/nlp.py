# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:06:07 2018

@author: ACER
"""

import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer 

data = pd.read_csv('dataset.csv', header=0)
data.head(5)



def preprocessing(line):
    line = line.lower()
    line = re.sub("[0-9]"," ",line) 
    tokenizer = RegexpTokenizer(r'\w+') 
    tokens = tokenizer.tokenize(line)

data['KomenBaru'] = data['Komentar'].apply(preprocessing)

X_train, X_test, y_train, y_test = train_test_split(data['KomenBaru'].values, data['Sentimen'].values, test_size=0.2,  random_state=42)

vec = CountVectorizer()
bow=vec.fit(data['KomenBaru'].values)
trainX = vec.transform(X_train)
testX = vec.transform(X_test)
#Naive Bayes
clf = MultinomialNB(alpha=0.5)
clf.fit(trainX,y_train)

tes = ["makanan mahal"]

X_t = vec.transform(tes)
hasil1 = clf.predict(X_t)
if hasil1 == 0:
    hasil2 = "Negatif"
else:
    hasil2 = "Positif"
print(tes, hasil1, hasil2)