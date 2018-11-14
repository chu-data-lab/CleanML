from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import pandas as pd

def drop_variables(X_train, X_test_list, drop_columns):
    n_test_files = len(X_test_list)
    X_train.drop(columns=drop_columns, inplace=True)
    for i in range(n_test_files):
        X_test_list[i].drop(columns=drop_columns, inplace=True)

def down_sample(X, y, random_state):
    rus = RandomUnderSampler(random_state=random_state)
    X_rus, y_rus = rus.fit_sample(X, y)
    indices = rus.sample_indices_
    X_train = X.iloc[indices, :]
    y_train = y.iloc[indices]
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
    x_train = vectorizer.fit_transform(corpus_train)
    x_test_list = [vectorizer.transform(corpus_test) for corpus_test in corpus_test_list]
    feature_names = vectorizer.get_feature_names()

    ch2 = SelectKBest(chi2, k=2000)
    x_train = ch2.fit_transform(x_train, y_train)
    x_test_list = [ch2.transform(x_test) for x_test in x_test_list]
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

def preprocess(dataset, X_train, y_train, X_test_list, y_test_list, normalize, down_sample_seed=1):
    if "drop_variables" in dataset.keys():
        drop_columns = dataset['drop_variables']
        drop_variables(X_train, X_test_list, drop_columns)

    if "class_imbalance" in dataset.keys() and dataset["class_imbalance"]:
        X_train, y_train = down_sample(X_train, y_train, down_sample_seed)

    if dataset['ml_task'] == 'classification':
        y_train, y_test_list = encode_cat_label(y_train, y_test_list)

    if "text_variables" in dataset.keys():
        text_columns = dataset["text_variables"]
        X_train, X_test_list = encode_text_features(X_train, X_test_list, y_train, text_columns)

    X_train, X_test_list = encode_cat_features(X_train, X_test_list)
    
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_list = [scaler.transform(X_test) for X_test in X_test_list]

    return X_train, y_train, X_test_list, y_test_list

