import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import code.Util.data_util as data_util


def run_svc(data_frame, enable_scaler):
    x = data_frame.drop(['salary'], axis=1)
    y = data_frame['salary']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
    print(f"X train:{x_train.shape} X test:{x_test.shape} Y train:{y_train.shape} Y test:{y_test.shape}")
    if enable_scaler:
        scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x)
        x_train = scaling.transform(x_train)
        x_test = scaling.transform(x_test)
    print(x_train)
    svc = SVC(class_weight='balanced', kernel='rbf', C=1, degree=2, gamma=0.1)
    cv = cross_val_score(svc, x_train, y_train, cv=10)
    model = svc.fit(x_train, y_train)
    score = svc.score(x_test, y_test)
    print("Cross Validation ", np.mean(cross_val_score(svc, x_train, y_train, cv=10)))
    print("Accuracy ", score)
    y_pred = svc.predict(x_test)
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
    print(cm)


# The algorithm is too slow to run with the full data set on the machine i own
# I used the first 5000 with only label Encoding
print("SVC model using all features (Label Encoding)")
print("No Scaler")
run_svc(data_util.get_label_encoded_data().loc[0:5000, :], True)
print("Scaler")
run_svc(data_util.get_label_encoded_data().loc[0:5000, :], False)
print("\nSVC model using some features (Label Encoding - model only uses age, education, occupation, hours-per-week)")
print("No Scaler")
run_svc(data_util.get_label_encoded_data_min().loc[0:5000, :], True)
print("Scaler")
run_svc(data_util.get_label_encoded_data_min().loc[0:5000, :], False)
