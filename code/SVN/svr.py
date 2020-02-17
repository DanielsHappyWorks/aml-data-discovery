import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix

boston = load_boston()


x, y = boston.data, boston.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.3, random_state=0)


#Scale feature values to between -1 and 1 but retain spacial orientation of data
scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train)

#apply scaling to both train and test set
x_train = scaling.transform(x_train)
x_test = scaling.transform(x_test)


#Our support vector regresors C is for margin softness/hardness
# svr = SVR(kernel='linear', C=1)
svr = SVR(kernel="poly", C=10, degree=2)
#svr = SVR(kernel="rbf", C=10, degree=2, gamma=0.1)

#Cross Validation with 10 folds
cv = cross_val_score(svr, x_train, y_train, cv=10)

#Create model with training data
model = svr.fit(x_train, y_train)

#get the score
score = svr.score(x_test, y_test)

print(" CV score ", np.mean(cv))

print("Acc", score)