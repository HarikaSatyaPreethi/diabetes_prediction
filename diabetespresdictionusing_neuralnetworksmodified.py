from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from numpy import array
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
np.random.seed(12)
dataset = np.loadtxt("/home/preethi/Documents/diabetis_Prediction/pima-indians-diabetes.data.csv", delimiter=",")
train,test=train_test_split(dataset,test_size=0.2)
train_feat=train[:,:8]
train_targ=train[:,8]
test_feat=test[:,:8]
test_targ=test[:,8]
print(train.shape)
print(test.shape)
model = Sequential()
model.add(Dense(40,input_dim=8, activation="relu"))
model.add(Dense(20, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_feat,train_targ,epochs=700, batch_size=10)
scores_train = model.evaluate(train_feat,train_targ)
scores_test = model.evaluate(test_feat,test_targ)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))
'''for i in range(len(test)):
	print(test[i])
y=model.predict_classes(test_feat)
print(y)'''
x=array([[ 8,95,72,0,0,36.8,0.485 ,57 ]])
y=model.predict_classes(x)
print(y)
