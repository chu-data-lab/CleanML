""" Load and preprocess data"""
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import pandas as pd
import utils

def check_version(dataset, error_type, train_file):
    """Check whether train and test are of the same version"""
    train_path_pfx = utils.get_dir(dataset, error_type, train_file)
    train_version = utils.get_version(train_path_pfx)
    test_files = utils.get_test_files(error_type, train_file)
    for test_file in test_files:
        test_path_pfx = utils.get_dir(dataset, error_type, test_file)
        test_version = utils.get_version(test_path_pfx)
        assert(train_version == test_version)
        
def load_data(dataset, train_path, test_path_list):
    """Load and split data into features and label.

    Args: 
        dataset (dict): dataset dict in config
        train_path (string): path for training set
        test_path_list (list): a list of paths for test set (missing values and outlier have multiple test sets)
    """
    # load data 
    train = utils.load_df(dataset, train_path)
    test_list = [utils.load_df(dataset, test_dir) for test_dir in test_path_list]
    
    # split X, y
    label = dataset['label']
    features = [v for v in train.columns if not v == label]
    X_train, y_train = train.loc[:, features], train.loc[:, label]
    X_test_list = [test.loc[:, features] for test in test_list]
    y_test_list = [test.loc[:, label] for test in test_list]

    return X_train, y_train, X_test_list, y_test_list  

def drop_variables(X_train, X_test_list, drop_columns):
    """Drop irrelavant features"""
    n_test_files = len(X_test_list)
    X_train.drop(columns=drop_columns, inplace=True)
    for i in range(n_test_files):
        X_test_list[i].drop(columns=drop_columns, inplace=True)

def down_sample(X, y, random_state):
    rus = RandomUnderSampler(random_state=random_state)
    X_rus, y_rus = rus.fit_sample(X, y)
    indices = rus.sample_indices_
    X_train = X.iloc[indices, :].reset_index(drop=True)
    y_train = y.iloc[indices].reset_index(drop=True)
    return X_train, y_train

def encode_cat_label(y_train, y_test_list):
    n_tr = y_train.shape[0]
    n_te_list = [y_test.shape[0] for y_test in y_test_list]
    test_split = np.cumsum(n_te_list)[:-1]

    y = pd.concat([y_train, *y_test_list], axis=0)
    le = LabelEncoder()
    y = le.fit_transform(y.values)

    y_train = y[:n_tr]
    y_test_list = np.split(y[n_tr:], test_split)
    return y_train, y_test_list

def text_embedding(corpus_train, corpus_test_list, y_train):
    vectorizer = TfidfVectorizer(stop_words='english')
    x_train_raw = vectorizer.fit_transform(corpus_train)
    x_test_list_raw = [vectorizer.transform(corpus_test) for corpus_test in corpus_test_list]
    feature_names = vectorizer.get_feature_names()

    k = min(200, x_train_raw.shape[1])
    ch2 = SelectKBest(chi2, k=k)
    x_train = ch2.fit_transform(x_train_raw, y_train)
    x_test_list = [ch2.transform(x_test) for x_test in x_test_list_raw]
    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    x_train = pd.DataFrame(x_train.toarray(), columns=feature_names)
    x_test_list = [pd.DataFrame(x_test.toarray(), columns=feature_names) for x_test in x_test_list]
    return x_train, x_test_list

def encode_text_features(X_train, X_test_list, y_train, text_columns):
    n_test_files = len(X_test_list)
    text_train = pd.DataFrame(X_train, columns=text_columns)
    text_test_list = [pd.DataFrame(X_test, columns=text_columns) for X_test in X_test_list]
    X_train.drop(columns=text_columns, inplace=True)
    for i in range(n_test_files):
        X_test_list[i].drop(columns=text_columns, inplace=True)

    for tc in text_columns:
        corpus_train = text_train.loc[:, tc]
        corpus_test_list = [text_test.loc[:, tc] for text_test in text_test_list]
        x_train, x_test_list = text_embedding(corpus_train, corpus_test_list, y_train)
        X_train = pd.concat([X_train, x_train], axis=1)
        for i in range(n_test_files):
            X_test_list[i] = pd.concat([X_test_list[i], x_test_list[i]], axis=1)
    return X_train, X_test_list

def encode_cat_features(X_train, X_test_list):
    n_tr = X_train.shape[0]
    n_te_list = [X_test.shape[0] for X_test in X_test_list]
    test_split = np.cumsum(n_te_list)[:-1]

    X = pd.concat([X_train, *X_test_list], axis=0)
    X = pd.get_dummies(X, drop_first=True).values.astype(float)

    X_train = X[:n_tr, :]
    X_test_list = np.split(X[n_tr:], test_split)
    return X_train, X_test_list

def preprocess(dataset, error_type, train_file, normalize=True, down_sample_seed=1):
    """Load and preprocess data

    Args:
        dataset (dict): dataset dict in config
        error_type (string): error type
        train_file (string): prefix of file of training set
        normalize (bool): whehter to standarize the data
        down_sample_seed: seed for down sampling
    """
    # check train and test version are consistent
    check_version(dataset, error_type, train_file)

    # get path of train file and test files
    train_path = utils.get_dir(dataset, error_type, train_file + "_train.csv")
    test_files = utils.get_test_files(error_type, train_file)
    test_path_list = [utils.get_dir(dataset, error_type, test_file + "_test.csv") for test_file in test_files]
    
    # load data
    X_train, y_train, X_test_list, y_test_list = load_data(dataset, train_path, test_path_list)

    ## preprocess data
    # drop irrelavant features
    if "drop_variables" in dataset.keys():
        drop_columns = dataset['drop_variables']
        drop_variables(X_train, X_test_list, drop_columns)

    # down sample if imbalanced
    if "class_imbalance" in dataset.keys() and dataset["class_imbalance"]:
        X_train, y_train = down_sample(X_train, y_train, down_sample_seed)

    # encode label
    if dataset['ml_task'] == 'classification':
        y_train, y_test_list = encode_cat_label(y_train, y_test_list)

    # text embedding
    if "text_variables" in dataset.keys():
        text_columns = dataset["text_variables"]
        X_train, X_test_list = encode_text_features(X_train, X_test_list, y_train, text_columns)

    # encode categorical features
    X_train, X_test_list = encode_cat_features(X_train, X_test_list)
    
    # normalize data
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_list = [scaler.transform(X_test) for X_test in X_test_list]

    return X_train, y_train, X_test_list, y_test_list, test_files

