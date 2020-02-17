import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_iris

iris = load_iris()

x = iris["data"][:,(2,3)]
y = (iris["target"] == 2).astype(np.float64)


print(y)


svm = Pipeline((("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss='hinge')),))

svm.fit(x,y)

print(svm.predict([[5.5,1.7]]))

print(svm.score(x,y))

y_pred = svm.predict(x)

cm = pd.DataFrame(confusion_matrix(y, y_pred))

print(cm)


