import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import code.Util.data_util as data_util

def run_svr(data_frame):
    x = data_frame.drop(['salary'], axis=1)
    y = data_frame['salary']
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.3, random_state=0)
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)
    print(f"X train:{x_train.shape} X test:{x_test.shape} Y train:{y_train.shape} Y test:{y_test.shape}")
    svr = SVR(kernel="rbf", C=10, degree=2, gamma=0.3)
    cv = cross_val_score(svr, x_train, y_train, cv=10)
    model = svr.fit(x_train, y_train)
    score = svr.score(x_test, y_test)
    print(" CV score ", np.mean(cv))
    print("Acc", score)

# Subset of dataset used as this is very slow ond overall performs very badly no matter the configuration for the dataset
print("SVC model using all features (Label Encoding)")
run_svr(data_util.get_label_encoded_data().loc[0:5000, :])
print("\nSVC model using some features (Label Encoding - model only uses age, education, occupation, hours-per-week)")
run_svr(data_util.get_label_encoded_data_min().loc[0:5000, :])
