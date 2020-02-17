import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston


boston = load_boston()


x = boston.data
y = boston.target

cats = pd.qcut(y, 5)

print(cats)


y_df = pd.DataFrame(y, columns={"nums"})


y_df.loc[(y_df["nums"] < 10), "cats"] = "low"
y_df.loc[(y_df["nums"] > 20), "cats"] = "high"
y_df.loc[(y_df["nums"] >= 10) & (y_df["nums"] <= 20), "cats"] = "med"



print(y_df)



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.33, random_state=32)


model = KNeighborsRegressor(n_neighbors=7)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print ("Cont score", score)


x_train, x_test, y_train, y_test = train_test_split(x,y_df["cats"],test_size=.33, random_state=32)

print(y_df["cats"].value_counts()/len(y_df["cats"]))

print(y_train.value_counts()/len(y_train))

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print ("Cat score", score)