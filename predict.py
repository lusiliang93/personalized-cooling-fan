import requests
import time
import numpy as np
import pickle
from sklearn import datasets, svm


filename = 'rf_model3.sav'
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
	print(t,h,s)
	# example
	# features: temperature, humidity, skin
	test = []
	#test = np.array([[-1.785714,-0.457604,-2.04924]])
	test.append(t)
	test.append(h)
	test.append(s)
	pred = load_model.predict(np.array([test]))
	print(pred)

	if (pred[0] > 0):
		os.system('wemo switch "WeMo Insight" on')
	else:
		os.system('wemo switch "WeMo Insight" off')

	# sleep for 3 seconds
	time.sleep(3)



