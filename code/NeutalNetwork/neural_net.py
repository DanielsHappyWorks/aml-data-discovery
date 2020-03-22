from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import code.Util.data_util as data_util


def run_nn_reg(data_frame):
    # NN Regressor
    x = data_frame.drop(['salary'], axis=1)
    y = data_frame['salary']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=32)
    print(f"X train:{x_train.shape} X test:{x_test.shape} Y train:{y_train.shape} Y test:{y_test.shape}")
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)
    nn = MLPRegressor(max_iter=300, hidden_layer_sizes=50, shuffle=True, activation='relu', solver='lbfgs')
    nn.fit(x_train, y_train)
    print("Score", nn.score(x_test, y_test))

def run_nn_class(data_frame):
    # NN Clasifier
    x = data_frame.drop(['salary'], axis=1)
    y = data_frame['salary']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=32)
    print(f"X train:{x_train.shape} X test:{x_test.shape} Y train:{y_train.shape} Y test:{y_test.shape}")
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)
    nn = MLPClassifier(max_iter=300, hidden_layer_sizes=50, shuffle=True, activation='relu', solver='lbfgs')
    nn.fit(x_train, y_train)
    print("Score", nn.score(x_test, y_test))

print("NN Regressor model using all features (Label Encoding)")
print("Regressor")
run_nn_reg(data_util.get_label_encoded_data())
print("Clasifier")
run_nn_class(data_util.get_label_encoded_data())
print("\nNN Regressor model using some features (Label Encoding - model only uses age, education, occupation, hours-per-week)")
print("Regressor")
run_nn_reg(data_util.get_label_encoded_data_min())
print("Clasifier")
run_nn_class(data_util.get_label_encoded_data_min())
