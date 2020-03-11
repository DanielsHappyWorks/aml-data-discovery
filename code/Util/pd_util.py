from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def do_label_encoding(data_frame, col):
    lb_encoder = LabelEncoder()
    data_frame[col] = lb_encoder.fit_transform(data_frame[col])
    return data_frame


def do_one_hot_encoding(data_frame, col):
    ohe_encoder = OneHotEncoder(sparse=False)
    transformed = ohe_encoder.fit_transform(data_frame[[col]])
    ohe_df = pd.DataFrame(transformed, columns=[w.replace('x0', col) for w in ohe_encoder.get_feature_names()])
    return pd.concat([data_frame, ohe_df], axis=1).drop([col], axis=1)
