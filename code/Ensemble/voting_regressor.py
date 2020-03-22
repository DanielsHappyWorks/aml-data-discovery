from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, VotingRegressor
import code.Util.data_util as data_util
from sklearn.neural_network import MLPRegressor, MLPClassifier

def run_voting_regressor(data_frame):
    x = data_frame.drop(['salary'], axis=1)
    y = data_frame['salary']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=32)
    print(f"X train:{x_train.shape} X test:{x_test.shape} Y train:{y_train.shape} Y test:{y_test.shape}")
    nn = MLPRegressor(max_iter=300, hidden_layer_sizes=50, activation='relu', solver='lbfgs')
    knn = KNeighborsRegressor(n_neighbors=147)
    lr = LinearRegression()
    voting_reg = VotingRegressor(estimators=[('nn', nn), ('knn', knn), ('lr', lr)])
    for reg in (nn, knn, lr, voting_reg):
        reg.fit(x_train, y_train)
        y_pred = reg.predict(x_test)
        print(reg.__class__.__name__, r2_score(y_test, y_pred))

def run_voting_classifier(data_frame):
    x = data_frame.drop(['salary'], axis=1)
    y = data_frame['salary']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=32)
    print(f"X train:{x_train.shape} X test:{x_test.shape} Y train:{y_train.shape} Y test:{y_test.shape}")
    nn = MLPClassifier(max_iter=300, hidden_layer_sizes=50, shuffle=True, activation='relu', solver='lbfgs')
    knn = KNeighborsClassifier(n_neighbors=147)
    gnb = GaussianNB()
    voting_reg = VotingClassifier(estimators=[('lr', nn), ('rf', knn), ('gnb', gnb)])
    for reg in (nn, knn, gnb, voting_reg):
        reg.fit(x_train, y_train)
        y_pred = reg.predict(x_test)
        print(reg.__class__.__name__, r2_score(y_test, y_pred))

print("NN Regressor model using all features (Label Encoding)")
print("Regressor")
run_voting_regressor(data_util.get_label_encoded_data())
print("Clasifier")
run_voting_classifier(data_util.get_label_encoded_data())
print("\nNN Regressor model using some features (Label Encoding - model only uses age, education, occupation, hours-per-week)")
print("Regressor")
run_voting_regressor(data_util.get_label_encoded_data_min())
print("Clasifier")
run_voting_classifier(data_util.get_label_encoded_data_min())


