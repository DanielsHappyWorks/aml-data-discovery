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

# encode using label encoding for salary as this is what were trying to predict
data = pd_util.do_label_encoding(data, 'salary')

# encode using one hot encoding (only encoding categorical data)
data = pd_util.do_one_hot_encoding(data, 'occupation')
data = pd_util.do_one_hot_encoding(data, 'education')

# remove unused columns for model (model only uses age, education, occupation, hours-per-week)
x = data.drop(
    ['salary', 'workclass', 'fnlwgt', 'education-num', 'marital-status', 'relationship', 'race', 'sex', 'capital-gain',
     'capital-loss', 'native-country'], axis=1)
y = data[['salary']]

print(x.head())
print(x.keys())
print(x.shape)
print(x.describe())

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=8)

print(f"X train:{X_train.shape} X test:{X_test.shape} Y train:{Y_train.shape} Y test:{Y_test.shape}")

# create model and do prediction
model = LinearRegression()
model.fit(X_train, Y_train)
Y_Predictions = model.predict(X_test)

print(f"rmse: {np.sqrt(mean_squared_error(Y_test, Y_Predictions))}")
print(f"r2: {r2_score(Y_test, Y_Predictions)}")