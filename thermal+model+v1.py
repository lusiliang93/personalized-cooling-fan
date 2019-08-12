
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


# In[4]:


data = pd.read_csv('lili.csv', sep=',',engine='python')
df = data


# In[5]:


# remove nan row
isnan = df['skin'].apply(np.isnan)
notnan = np.invert(isnan)
index = df['skin'].index[notnan]
data_new = data.iloc[index]
# get features and thermal sensation
y = data_new['sensation']
x = data_new[['temperature','humidity','skin','clothing']]


# In[6]:


import sklearn.preprocessing, sklearn.decomposition,sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import datasets, svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn_pandas import DataFrameMapper


# In[7]:


mapper = DataFrameMapper([(['temperature'], None),
                         (['humidity'], None),
                          (['skin'], None),
                          (['clothing'], None)])
mapper.fit_transform(x.copy())
# count the number of thermal sensation
bool = (y==3)
len(y[bool])


# In[21]:


#clf = svm.SVC(kernel='linear')
C=1
#clf = svm.SVC(kernel='poly',degree=3,C=C)
clf = svm.SVC(kernel='rbf',gamma=0.7,C=C)
pipe = sklearn.pipeline.Pipeline([('featurize', mapper),('svc', clf)])
#np.round(cross_val_score(pipe, X=data_new.copy(), y=data_new.comfort, scoring='r2'), 2)
cross_val_score(pipe, X=x.copy(), y=y, scoring='r2',cv=5)


# In[23]:


# testing
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


# In[26]:


predicted = cross_val_predict(clf, x.copy(), y, cv=5)
metrics.accuracy_score(y, predicted) 


# In[30]:


test = x.iloc[[0]]
clf.fit(x,y)
gt = y[0]
pred = clf.predict(test)
pred
gt

