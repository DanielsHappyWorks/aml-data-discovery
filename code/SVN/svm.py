import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_iris
import code.Util.data_util as data_util

def run_linear_svc(data_frame):
    iris = load_iris()
    x = iris["data"][:,(2,3)]
    y = (iris["target"] == 2).astype(np.float64)
    print(y)
    svm = Pipeline((("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss='hinge')),))
    svm.fit(x,y)
    print(svm.predict([[5.5,1.7]]))
    print("Accuracy ", svm.score(x,y))
    y_pred = svm.predict(x)
    cm = pd.DataFrame(confusion_matrix(y, y_pred))
    print(cm)

# The algorithm is too slow to run with the full data set on the machine i own
# I used the first 5000 with only label Encoding
print("SVM model using all features (Label Encoding)")
run_linear_svc(data_util.get_label_encoded_data().loc[0:5000, :])
print("\nSVM model using some features (Label Encoding - model only uses age, education, occupation, hours-per-week)")
run_linear_svc(data_util.get_label_encoded_data_min().loc[0:5000, :])

