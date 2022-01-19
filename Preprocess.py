# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:42:02 2020

@author: Nandini Laad
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset=pd.read_csv('Valid.tsv',error_bad_lines=False, delimiter='\t', quoting=3, encoding='latin-1')

#Cleaning the datasets
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,4995):
    review=re.sub('[^a-zA-Z]',' ',dataset['text'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

#Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=12000)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

#Splitting dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0);

#Training the model
#from sklearn.naive_bayes import GaussianNB
#classifier=GaussianNB()
#classifier.fit(X_train,y_train)

#from sklearn.naive_bayes import MultinomialNB
#classifier=MultinomialNB()
#classifier.fit(X_train,y_train)

#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
#classifier.fit(X_train,y_train)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

#from sklearn.svm import SVC
#classifier = SVC(kernel='rbf',random_state=0)
#classifier.fit(X_train,y_train)

#from sklearn import svm
#classifier = svm.SVC()
#classifier.fit(X_train,y_train)
#Testing the model
y_pred=classifier.predict(X_test)

#Making of confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
ac=accuracy_score(y_test,y_pred)
print(ac)