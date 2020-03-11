import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import code.Util.data_util as data_util


def run_knn(data_frame):
    x = data_frame.drop(['salary'], axis=1)

    # Regressor knn model
    x_train, x_test, y_train, y_test = train_test_split(x, data_frame[['salary']], test_size=.33, random_state=1)
    print(f"k=sqrt(n) = {np.sqrt(len(x_train.index))}")
    model = KNeighborsRegressor(n_neighbors=147)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print("Regressor Model score", score)

    # Classifier knn model
    x_train, x_test, y_train, y_test = train_test_split(x, data_frame["salary"], test_size=.33, random_state=1)
    model = KNeighborsClassifier(n_neighbors=147)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print("Classifier Model score", score)


print("KNN model using all features (Label Encoding)")
run_knn(data_util.get_label_encoded_data())
print("KNN model using all features (One Hot Encoding)")
run_knn(data_util.get_one_hot_encoded_data())
print("KNN model using some features (Label Encoding - model only uses age, education, occupation, hours-per-week)")
run_knn(data_util.get_label_encoded_data_min())
print("KNN model using some features (One Hot Encoding - model only uses age, education, occupation, hours-per-week)")
run_knn(data_util.get_one_hot_encoded_data_min())
