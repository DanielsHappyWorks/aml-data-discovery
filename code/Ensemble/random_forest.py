from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import code.Util.data_util as data_util

def run_rf_regressor(data_frame):
    x = data_frame.drop(['salary'], axis=1)
    y = data_frame['salary']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=32)
    print(f"X train:{x_train.shape} X test:{x_test.shape} Y train:{y_train.shape} Y test:{y_test.shape}")
    model = RandomForestRegressor(n_estimators=10, n_jobs=-1)
    model.fit(x_train, y_train)
    print("Score ", model.score(x_test, y_test))
    for name, score in zip(data_frame.columns, model.feature_importances_):
        print(name, score)

def run_rf_classifier(data_frame):
    x = data_frame.drop(['salary'], axis=1)
    y = data_frame['salary']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=32)
    print(f"X train:{x_train.shape} X test:{x_test.shape} Y train:{y_train.shape} Y test:{y_test.shape}")
    model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    model.fit(x_train, y_train)
    print("Score ", model.score(x_test, y_test))
    for name, score in zip(data_frame.columns, model.feature_importances_):
        print(name, score)

print("Random Forest Regressor")
run_rf_regressor(data_util.get_label_encoded_data())
print("\nRandom Forest Classifier ")
run_rf_classifier(data_util.get_label_encoded_data())
print("\nRandom Forest Regressor (Label Encoding - model only uses age, education, occupation, hours-per-week)")
run_rf_regressor(data_util.get_label_encoded_data_min())
print("\nRandom Forest Classifier (Label Encoding - model only uses age, education, occupation, hours-per-week)")
run_rf_classifier(data_util.get_label_encoded_data_min())
