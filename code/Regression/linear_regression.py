import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("../dataset/adult.data")
data[data.isnull().any(axis=1)]

print(data.head())
print(data.keys())
print(data.shape)
print(data.describe())

lb_make = LabelEncoder()
ohe = OneHotEncoder(sparse=False)
data['salary'] = lb_make.fit_transform(data['salary'])

transformed = ohe.fit_transform(data[['occupation']])
ohe_df = pd.DataFrame(transformed, columns=[w.replace('x0', 'occupation') for w in ohe.get_feature_names()])
data = pd.concat([data, ohe_df], axis=1).drop(['occupation'], axis=1)

x = data[['age']]
x = data.drop(['salary', 'workclass', 'fnlwgt', 'education', 'marital-status', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','native-country'], axis = 1)
y = data[['salary']]

print(x.head())
print(x.keys())
print(x.shape)
print(x.describe())

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.33, random_state = 8)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


model = LinearRegression()

model.fit(X_train, Y_train)

Y_Pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test,Y_Pred))

r2 = r2_score(Y_test,Y_Pred)

print ('rmse = ', rmse)
print ('r2 ', r2)



