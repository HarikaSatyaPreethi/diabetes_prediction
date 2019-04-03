import pandas as pd
import numpy as np
from numpy import array
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
pima=pd.read_csv("/home/preethi/Documents/diabetis_Prediction/pima-indians-diabetes.data.csv")
lr=LogisticRegression()
pima.head(3)
pima.tail(3)
pima.info()
pima.shape

train,test=train_test_split(pima,test_size=0.2)
test.shape
test.shape
train_feat=train.iloc[:,:8]
train_targ=train.iloc[:,8]
train_feat.shape
train_targ.shape

lr.fit(train_feat,train_targ)
lr.score(train_feat,train_targ)
test_feat=test.iloc[:,:8]
test_targ=test.iloc[:,8]
lr.score(test_feat,test_targ)
x=array([[8,183,64,0,0,23.3,0.672,32]])
print(x)

y=lr.predict(x)
print(y)
lr.predict(test_feat)
confusion_matrix(lr.predict(train_feat),train_targ)
confusion_matrix(lr.predict(test_feat),test_targ)
lr.predict(train_feat)
