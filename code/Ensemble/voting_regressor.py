import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from sklearn.ensemble import VotingRegressor

boston = load_boston()


x, y = pd.DataFrame(boston.data), boston.target

x.columns = boston.feature_names
print(x)

x = x['RM'][:,np.newaxis]



x_train, x_test, y_train, y_test =  train_test_split(x,y,test_size=.3, random_state=32)

lin_reg = LinearRegression()
knn_reg = KNeighborsRegressor()
svr_reg = SVR()


voting_reg = VotingRegressor(estimators=[('lr', lin_reg), ('knn', knn_reg), ('svr', svr_reg)])

for reg in (lin_reg, knn_reg, svr_reg, voting_reg):
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    print(reg.__class__.__name__, r2_score(y_test,y_pred))


