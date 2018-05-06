import requests
import time
import numpy as np
import pickle
from sklearn import datasets, svm


filename = 'rf_model.sav'
load_model = pickle.load(open(filename,'rb'))

while (True):
	# get parameters from particle cloud
	humidity = requests.get('https://api.particle.io/v1/devices/28003a000c47343438323536/humidity?access_token=8608ee061d524b0682c1880998688651a14b73de')
	#print(r.json()['result'])
	h = humidity.json()['result']
	temperature = requests.get('https://api.particle.io/v1/devices/28003a000c47343438323536/temperature?access_token=8608ee061d524b0682c1880998688651a14b73de')
	t = temperature.json()['result']
	skin = requests.get('https://api.particle.io/v1/devices/28003a000c47343438323536/skin?access_token=8608ee061d524b0682c1880998688651a14b73de')
	s = skin.json()['result']
	print(h,t,s)
	# example
	# features: temperature, humidity, skin
	test = np.zeros(3)
	#test = np.array([[-1.785714,-0.457604,-2.04924,21.7]]).
	test[0] = t
	test[1] = h
	test[2] = s
	pred = load_model.predict(test)
	print pred

	if (pred[0] > 0):
		os.system('wemo switch "WeMo Insight" on')
	else:
		os.system('wemo switch "WeMo Insight" off')

	# sleep for 3 seconds
	time.sleep(3)



