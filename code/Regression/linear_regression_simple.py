from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import numpy as np


x = np.array([0,1,2,3,4,5])
y = np.array([0,0.9,0.8,0.1,-0.4,-0.8])

x = x[:,np.newaxis]
y = y[:,np.newaxis]



plt.scatter(x,y)

plt.show()

model = LinearRegression()

model.fit(x,y)

yPred = model.predict(x)

print(y)
print(yPred)

rmse = np.sqrt(mean_squared_error(y,yPred))

r2 = r2_score(y,yPred)

print ('rmse = ', rmse)
print ('r2 ', r2)

plt.scatter(x,y)
plt.plot(x,yPred, 'r-')

plt.show()