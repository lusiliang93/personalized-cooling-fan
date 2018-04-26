import requests
import time

sns.set(color_codes=True)

filename = 'svm_model.sav'
load_model = pickle.load(open(filename,'rb'))

while (True):
	# get parameters from particle cloud
	r = requests.get('https://api.particle.io/v1/devices/1f0028000c47343438323536/test?access_token=062530ccee58bc2babf77e8d1bc8500e4ac44014')
	print r.json()['result']
	# example
	# features: temperature_normal, humidity_normal, skin_normal, clothing
	test = np.array([[-1.785714,-0.457604,-2.04924,21.7]])
	pred = load_model.predict(test)


	# sleep for 3 seconds
	time.sleep(3)



