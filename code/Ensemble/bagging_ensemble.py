from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
import code.Util.data_util as data_util

def run_bagging_regressor(data_frame):
    x = data_frame.drop(['salary'], axis=1)
    y = data_frame['salary']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=32)
    print(f"X train:{x_train.shape} X test:{x_test.shape} Y train:{y_train.shape} Y test:{y_test.shape}")
    bagging_reg = BaggingRegressor(n_estimators=10, max_samples=50, bootstrap=True, n_jobs=-1,
                                   random_state=20)
    bagging_reg.fit(x_train, y_train)
    y_pred = bagging_reg.predict(x_test)
    print('Accuracy ', r2_score(y_test, y_pred))

def run_bagging_classifier(data_frame):
    x = data_frame.drop(['salary'], axis=1)
    y = data_frame['salary']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=32)
    print(f"X train:{x_train.shape} X test:{x_test.shape} Y train:{y_train.shape} Y test:{y_test.shape}")
    bagging_reg = BaggingClassifier(n_estimators=10, max_samples=50, bootstrap=True, n_jobs=-1,
                                   random_state=20)
    bagging_reg.fit(x_train, y_train)
    y_pred = bagging_reg.predict(x_test)
    print('Accuracy ', r2_score(y_test, y_pred))

print("Bagging Regressor")
run_bagging_regressor(data_util.get_label_encoded_data())
print("\nBagging Classifier ")
run_bagging_classifier(data_util.get_label_encoded_data())
print("\nBagging Regressor (Label Encoding - model only uses age, education, occupation, hours-per-week)")
run_bagging_regressor(data_util.get_label_encoded_data_min())
print("\nBagging Classifier (Label Encoding - model only uses age, education, occupation, hours-per-week)")
run_bagging_classifier(data_util.get_label_encoded_data_min())
