import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import code.Util.data_util as data_util
from sklearn.preprocessing import PolynomialFeatures


def run_regression(x_df, y_df, enable_poly):
    print("~~~~~~~~~~~~ Starting Regression ~~~~~~~~~~~~")
    X_train, X_test, Y_train, Y_test = train_test_split(x_df, y_df, test_size=0.33, random_state=8)

    print(f"X train:{X_train.shape} X test:{X_test.shape} Y train:{Y_train.shape} Y test:{Y_test.shape}")

    # create model and do prediction
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_Predictions = model.predict(X_test)

    print(f"rmse: {np.sqrt(mean_squared_error(Y_test, Y_Predictions))}")
    print(f"r2: {r2_score(Y_test, Y_Predictions)}")

    if enable_poly:
        # create model and do prediction (poly degree=2)
        poly_features = PolynomialFeatures(degree=2)
        model = LinearRegression()
        model.fit(poly_features.fit_transform(X_train), Y_train)
        Y_Predictions_Deg_2 = model.predict(poly_features.fit_transform(X_test))

        print(f"Poly (deg 2) rmse: {np.sqrt(mean_squared_error(Y_test, Y_Predictions_Deg_2))}")
        print(f"Poly (deg 2) r2: {r2_score(Y_test, Y_Predictions_Deg_2)}")


data = data_util.get_label_encoded_data_min()
print("~~~~~~~~~~~~Linear Regression on minimal data set using label encoding~~~~~~~~~~~~")
x = data.drop(['salary'], axis=1)
y = data[['salary']]
run_regression(x, y, True)

data = data_util.get_label_encoded_data()
print("~~~~~~~~~~~~Linear Regression on whole data set using label encoding~~~~~~~~~~~~")
x = data.drop(['salary'], axis=1)
y = data[['salary']]
run_regression(x, y, True)

# No polynomial output is generated as it is too slow with the amount of features available
data = data_util.get_one_hot_encoded_data_min()
print("~~~~~~~~~~~~Linear Regression on minimal data set using one hot encoding~~~~~~~~~~~~")
x = data.drop(['salary'], axis=1)
y = data[['salary']]
run_regression(x, y, False)

# No polynomial output is generated as it is too slow with the amount of features available
data = data_util.get_one_hot_encoded_data()
print("~~~~~~~~~~~~Linear Regression on whole data set using one hot encoding~~~~~~~~~~~~")
x = data.drop(['salary'], axis=1)
y = data[['salary']]
run_regression(x, y, False)
