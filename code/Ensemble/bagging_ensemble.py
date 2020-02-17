import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

from sklearn.ensemble import VotingRegressor, BaggingRegressor

boston = load_boston()


x, y = pd.DataFrame(boston.data), boston.target

x.columns = boston.feature_names
print(x)



x_train, x_test, y_train, y_test =  train_test_split(x,y,test_size=.3, random_state=32)


bagging_reg = BaggingRegressor(LinearRegression(), n_estimators=10, max_samples=50, bootstrap=True, n_jobs=-1, random_state=20)

bagging_reg.fit(x_train, y_train)

y_pred = bagging_reg.predict(x_test)

print('Accuracy ', r2_score(y_test, y_pred))