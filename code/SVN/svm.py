import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import code.Util.data_util as data_util
from sklearn.preprocessing import MinMaxScaler


def run_linear_svc(data_frame):
    x = data_frame.drop(['salary'], axis=1)
    y = data_frame['salary']
    svm = Pipeline((("scaler", MinMaxScaler(feature_range=(-1, 1))), ("linear_svc", LinearSVC(C=1, dual=False)),))
    svm.fit(x, y)
    print("Accuracy ", svm.score(x, y))
    y_pred = svm.predict(x)
    cm = pd.DataFrame(confusion_matrix(y, y_pred))
    print(cm)


print("SVM model using all features (Label Encoding)")
run_linear_svc(data_util.get_label_encoded_data())
print("\nSVM model using some features (Label Encoding - model only uses age, education, occupation, hours-per-week)")
run_linear_svc(data_util.get_label_encoded_data_min())
