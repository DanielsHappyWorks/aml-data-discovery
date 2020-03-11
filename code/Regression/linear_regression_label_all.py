import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import code.Util.pd_util as pd_util
from sklearn.preprocessing import PolynomialFeatures

# read data
data = pd.read_csv("../dataset/adult.data")
data[data.isnull().any(axis=1)]

# encode using label encoding (only encoding categorical data)
data = pd_util.do_label_encoding(data, 'salary')
data = pd_util.do_label_encoding(data, 'workclass')
data = pd_util.do_label_encoding(data, 'education')
data = pd_util.do_label_encoding(data, 'marital-status')
data = pd_util.do_label_encoding(data, 'occupation')
data = pd_util.do_label_encoding(data, 'relationship')
data = pd_util.do_label_encoding(data, 'race')
data = pd_util.do_label_encoding(data, 'sex')
data = pd_util.do_label_encoding(data, 'native-country')

x = data.drop(['salary'], axis=1)
y = data[['salary']]

print(x.head())
print(x.keys())
print(x.shape)
print(x.describe())

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=8)

print(f"X train:{X_train.shape} X test:{X_test.shape} Y train:{Y_train.shape} Y test:{Y_test.shape}")

# create model and do prediction (lr degree=1)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_Predictions = model.predict(X_test)

print(f"rmse: {np.sqrt(mean_squared_error(Y_test, Y_Predictions))}")
print(f"r2: {r2_score(Y_test, Y_Predictions)}")

# create model and do prediction (poly degree=2)
poly_features = PolynomialFeatures(degree=2)
model = LinearRegression()
model.fit(poly_features.fit_transform(X_train), Y_train)
Y_Predictions_Deg_2 = model.predict(poly_features.fit_transform(X_test))

print(f"rmse: {np.sqrt(mean_squared_error(Y_test, Y_Predictions_Deg_2))}")
print(f"r2: {r2_score(Y_test, Y_Predictions_Deg_2)}")
