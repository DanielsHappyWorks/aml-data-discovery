import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris

iris = load_iris()

rnd_clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)

rnd_clf.fit(iris.data, iris.target)

for name, score in zip(iris.feature_names, rnd_clf.feature_importances_):
    print(name,score)