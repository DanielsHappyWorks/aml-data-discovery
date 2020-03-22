import pandas as pd
import code.Util.pd_util as pd_util


def get_label_encoded_data():
    data = get_data()

    # encode using label encoding (only encoding categorical data)
    data = pd_util.do_label_encoding(data, 'salary')
    data = pd_util.do_label_encoding(data, 'workclass')
    data = pd_util.do_label_encoding(data, 'education')
    data = pd_util.do_label_encoding(data, 'marital-status')
    data = pd_util.do_label_encoding(data, 'occupation')
    data = pd_util.do_label_encoding(data, 'relationship')
    data = pd_util.do_label_encoding(data, 'race')
    data = pd_util.do_label_encoding(data, 'sex')
    data = pd_util.do_label_encoding(data, 'native-country')
    return data


def get_label_encoded_data_min():
    data = get_data()

    # encode using label encoding
    data = pd_util.do_label_encoding(data, 'salary')
    data = pd_util.do_label_encoding(data, 'occupation')
    data = pd_util.do_label_encoding(data, 'education')

    # remove unused columns for model (model only uses age, education, occupation, hours-per-week)
    data = data.drop(
        ['workclass', 'fnlwgt', 'education-num', 'marital-status', 'relationship', 'race', 'sex',
         'capital-gain',
         'capital-loss', 'native-country'], axis=1)
    return data


def get_one_hot_encoded_data():
    data = get_data()

    # encode using label encoding for salary as this is what were trying to predict
    data = pd_util.do_label_encoding(data, 'salary')

    # encode using one hot encoding (only encoding categorical data)
    data = pd_util.do_one_hot_encoding(data, 'workclass')
    data = pd_util.do_one_hot_encoding(data, 'education')
    data = pd_util.do_one_hot_encoding(data, 'marital-status')
    data = pd_util.do_one_hot_encoding(data, 'occupation')
    data = pd_util.do_one_hot_encoding(data, 'relationship')
    data = pd_util.do_one_hot_encoding(data, 'race')
    data = pd_util.do_one_hot_encoding(data, 'sex')
    data = pd_util.do_one_hot_encoding(data, 'native-country')
    return data


def get_one_hot_encoded_data_min():
    data = get_data()

    # encode using label encoding for salary as this is what were trying to predict
    data = pd_util.do_label_encoding(data, 'salary')

    # encode using one hot encoding (only encoding categorical data)
    data = pd_util.do_one_hot_encoding(data, 'occupation')
    data = pd_util.do_one_hot_encoding(data, 'education')

    # remove unused columns for model (model only uses age, education, occupation, hours-per-week)
    data = data.drop(
        ['workclass', 'fnlwgt', 'education-num', 'marital-status', 'relationship', 'race', 'sex',
         'capital-gain',
         'capital-loss', 'native-country'], axis=1)
    return data


def get_data():
    data = pd.read_csv("../dataset/adult.data")
    data[data.isnull().any(axis=1)]
    return data



