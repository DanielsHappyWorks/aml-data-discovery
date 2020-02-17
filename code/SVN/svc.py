import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt

digits = load_digits()

x, y = digits.data, digits.target

for i in range(10):
    print(i)
    plt.subplot(2, 5, i+1)
    plt.imshow(digits.images[i], cmap='binary', interpolation='none')

plt.show()

print(x[0].reshape(8,8))

uniques = np.unique(y, return_counts=True)

print(uniques)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)

scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train)

print(x_train)

x_train = scaling.transform(x_train)
x_test = scaling.transform(x_test)



print(x_train)


svc = SVC(class_weight='balanced', kernel='rbf', C=1, degree=2, gamma=0.1)

cv = cross_val_score(svc, x_train, y_train, cv=10)

model = svc.fit(x_train, y_train)

score = svc.score(x_test, y_test)

print ("Cross Validation ",np.mean(cv))
print ("Overall Accuracy ",score)


y_pred = svc.predict(x_test)

cm = pd.DataFrame(confusion_matrix(y_test, y_pred))

print(cm)