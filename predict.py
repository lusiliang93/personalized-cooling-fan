import requests
import time
import numpy as np
import pickle
from sklearn import datasets, svm
import os


filename = 'rf_model3.sav'
load_model = pickle.load(open(filename,'rb'))

while (True):
	# get parameters from particle cloud
	humidity = requests.get('https://api.particle.io/v1/devices/28003a000c47343438323536/humidity?access_token=8608ee061d524b0682c1880998688651a14b73de')
	h = humidity.json()['result']
	temperature = requests.get('https://api.particle.io/v1/devices/28003a000c47343438323536/temperature?access_token=8608ee061d524b0682c1880998688651a14b73de')
	t = temperature.json()['result']
	skin = requests.get('https://api.particle.io/v1/devices/28003a000c47343438323536/skin?access_token=8608ee061d524b0682c1880998688651a14b73de')
	s = skin.json()['result']
	print('temperature:',t)
	print('humidity:',h)
	print('skin temperature:',s)
	# example
	# features: temperature, humidity, skin
	test = []
	test.append(t)
	test.append(h)
	test.append(s)
	pred = load_model.predict(np.array([test]))
	# control logic
	temp = 0
	if (s > 32 or t > 30):
		temp = 2
	elif (s < 30):
		temp = -2
	else:
		temp = 0
	pred = (temp + pred[0]) / 2
	if (pred > 0):
		print('thermal sensation: warm')
	elif(pred < 0):
		print('thermal sensation: cold')
	else:
		print('thermal sensation: neutral')

	if (pred > 0):
		os.system('wemo switch "WeMo Insight" on')
	else:
		os.system('wemo switch "WeMo Insight" off')

	# sleep for 3 seconds
	time.sleep(3)



