import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt


digits = load_digits()

x, y = digits.data, digits.target

for i in range(10):
    print(i)
    plt.subplot(2, 5, i+1)
    plt.imshow(digits.images[i], cmap='binary', interpolation='none')

plt.show()


print(x[0].reshape(8,8))
print(y[0])

uniques = np.unique(y, return_counts=True)

print(uniques)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)

scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train)

print("x train before", x_train)

x_train = scaling.transform(x_train)
x_test = scaling.transform(x_test)

print("x train after",x_train)

svc = SVC( kernel='linear', C=10.0)

cv = cross_val_score(svc, x_train, y_train, cv=10)

model = svc.fit(x_train, y_train)

score = svc.score(x_test, y_test)

print ("Cross Validation ",np.mean(cv))
print ("SVC Alone Accuracy ",score)

y_pred = svc.predict(x_test)

cm = pd.DataFrame(confusion_matrix(y_test, y_pred))

print(cm)


ada_clf = AdaBoostClassifier(SVC( kernel='linear', C=.01), algorithm="SAMME", learning_rate=0.5, random_state=32, n_estimators=5)

ada_clf.fit(x_train, y_train)

score = ada_clf.score(x_test, y_test)

print("ADA Score ", score)

y_pred = ada_clf.predict(x_test)

cm = pd.DataFrame(confusion_matrix(y_test, y_pred))

print(cm)


