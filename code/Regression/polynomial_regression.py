from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import numpy as np


x = np.array([0,1,2,3,4,5])
y = np.array([0,0.9,0.8,0.1,-0.4,-0.8])

x = x[:,np.newaxis]
y = y[:,np.newaxis]



poly_features = PolynomialFeatures(degree=2)

xTrans = poly_features.fit_transform(x)


plt.scatter(x,y)

plt.show()

model = LinearRegression()
model2 = LinearRegression()

model.fit(x,y)

model2.fit(xTrans, y)

yPred = model.predict(x)

yPred2 = model2.predict(xTrans)

print(y)
print(yPred2)

rmse = np.sqrt(mean_squared_error(y,yPred))

r2 = r2_score(y,yPred)

print ('rmse = ', rmse)
print ('r2 ', r2)

rmse = np.sqrt(mean_squared_error(y,yPred2))

r2 = r2_score(y,yPred2)

print ('rmse = ', rmse)
print ('r2 ', r2)

plt.scatter(x,y)
plt.plot(x,yPred, 'r-')
plt.plot(x,yPred2, 'b--')

plt.show()