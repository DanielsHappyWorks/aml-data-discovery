from sklearn.cluster import KMeans
import code.Util.data_util as data_util
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

def run_kmeans(data_frame):
    x = data_frame.drop(['salary'], axis=1)
    y = data_frame[['salary']]
    model = KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    y_pred = model.fit_predict(x)
    print("Accuracy ", accuracy_score(y, y_pred))
    cm = pd.DataFrame(confusion_matrix(y, y_pred))
    print(cm)

print("kMeans model using all features (Label Encoding)")
run_kmeans(data_util.get_label_encoded_data())
print("\nkMeans model using all features (One Hot Encoding)")
run_kmeans(data_util.get_one_hot_encoded_data())
print("\nkMeans model using some features (Label Encoding - model only uses age, education, occupation, hours-per-week)")
run_kmeans(data_util.get_label_encoded_data_min())
print("\nkMeans model using some features (One Hot Encoding - model only uses age, education, occupation, hours-per-week)")
run_kmeans(data_util.get_one_hot_encoded_data_min())
