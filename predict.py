import requests
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing, sklearn.decomposition,sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import datasets, svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import time

sns.set(color_codes=True)


while (True):
	# get parameters from particle cloud
	r = requests.get('https://api.particle.io/v1/devices/1f0028000c47343438323536/test?access_token=062530ccee58bc2babf77e8d1bc8500e4ac44014')
	print r.json()['result']

	data = pd.read_csv('lili.csv', sep=',',engine='python')
	# remove nan row
	isnan = df['skin'].apply(np.isnan)
	notnan = np.invert(isnan)
	index = df['skin'].index[notnan]
	data_new = data.iloc[index]
	# get features and thermal sensation
	y = data_new['sensation']
	x = data_new[['temperature','humidity','skin','clothing']]

	mapper = DataFrameMapper([(['temperature'], None),
	                         (['humidity'], None),
	                          (['skin'], None),
	                          (['clothing'], None)])
	mapper.fit_transform(x.copy())
	# count the number of thermal sensation
	bool = (y==3)
	len(y[bool])

	clf = svm.SVC(kernel='linear')
	pipe = sklearn.pipeline.Pipeline([('featurize', mapper),('svc', clf)])
	#np.round(cross_val_score(pipe, X=data_new.copy(), y=data_new.comfort, scoring='r2'), 2)
	cross_val_score(pipe, X=x.copy(), y=y, scoring='r2',cv=5)

	# prediction
	predicted = cross_val_predict(clf, x.copy(), y, cv=5)
	metrics.accuracy_score(y, predicted) 

	test = x.iloc[[0]]
	clf.fit(x,y)
	gt = y[0]
	pred = clf.predict(test)
	pred
	test
	gt

	# sleep for 3 seconds
	time.sleep(3)



