
# coding: utf-8

# In[1]:


# Import required libraries

import numpy as np
import pandas as pd


# In[2]:


# Load the dataset
in_file = '868c0ef8-c-HEDatasetML/train.csv'
full_data = pd.read_csv(in_file)


# In[3]:


# Store the 'target' feature in a new variable and remove it from the dataset
targets = full_data['target']
features = full_data.drop('target', axis = 1)


# In[4]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(targets)
targetFinal = le.transform(targets)


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

complaintSummary = features['complaint_summary']
vectorizer = CountVectorizer()
complaintFeature = vectorizer.fit_transform(complaintSummary)

tfidf_transformer = TfidfTransformer()
featuresFinal = tfidf_transformer.fit_transform(complaintFeature)


# In[6]:


# Use a random forest classifier to learn and predict
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100,random_state=392)
model = model.fit(featuresFinal, targetFinal)


# In[7]:


# import the test file
test_file = '868c0ef8-c-HEDatasetML/test.csv'
testData = pd.read_csv(test_file)

testFeatures = testData['complaint_summary']
testComplaintFeature = vectorizer.transform(testFeatures)
testtfidf = tfidf_transformer.transform(testComplaintFeature)

predictions = model.predict(testtfidf)


# In[32]:


import csv

with open('868c0ef8-c-HEDatasetML/submit.csv', 'w', newline='') as csvfile:
    fieldnames = ['officer_id', 'target']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(predictions)):
        writer.writerow({'officer_id': testData['officer_id'][i], 'target': le.inverse_transform(predictions[i])})

