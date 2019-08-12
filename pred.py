import numpy as np
import pickle
from sklearn import datasets, svm


filename = 'svm_model.sav'
load_model = pickle.load(open(filename,'rb'))
# example
# features: temperature_normal, humidity_normal, skin_normal, clothing
test = np.array([[-1.785714,-0.457604,-2.04924,21.7]])
pred = load_model.predict(test)
print(pred)