import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

x,y = make_blobs(n_samples=3, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

boston = load_boston()

print(boston.feature_names)
df = pd.DataFrame(boston.data, columns=boston.feature_names)

df["PRICE"] = boston.target

print(df[["AGE", "PRICE"]])

x = df[["AGE", "PRICE"]].to_numpy()


print(x)

plt.scatter(x[:,0], x[:, 1], c="white", marker='o', edgecolors='black', s=50)

plt.show()


model = KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)


y_pred = model.fit_predict(x)

n_clusters_ = model.n_clusters


from itertools import cycle


plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = model.labels_ == k
    cluster_center = model.cluster_centers_[k]
    plt.plot(x[my_members, 0], x[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


distortions = []

for i in range(1,12):
    model = KMeans(n_clusters=i, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)

    model.fit(x)

    distortions.append(model.inertia_)

plt.plot(range(1, 12), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
