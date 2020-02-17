import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston

boston = load_boston()

print (boston.keys())
print (boston.data.shape)

print(boston.feature_names)

print (boston.DESCR)


bos = pd.DataFrame(boston.data)

print(bos.head())

bos.columns = boston.feature_names
print(bos.head())


print (boston.target.shape)


bos['PRICE'] = boston.target

print (bos.head())

print(bos.describe())


x= bos.drop('PRICE', axis=1)
y = bos['PRICE']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.33, random_state = 8)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


model = LinearRegression()

model.fit(X_train, Y_train)

Y_Pred = model.predict(X_test)


plt.scatter(Y_test, Y_Pred)

plt.show()


rmse = np.sqrt(mean_squared_error(Y_test,Y_Pred))

r2 = r2_score(Y_test,Y_Pred)

print ('rmse = ', rmse)
print ('r2 ', r2)



