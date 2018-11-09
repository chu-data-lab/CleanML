import numpy as np
import pandas as pd
import argparse
import config
import utils
import sys
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import pickle

def text_embedding(corpus_train, corpus_test_list, y_train):
    vectorizer = TfidfVectorizer(stop_words='english')
    x_train = vectorizer.fit_transform(corpus_train)
    x_test_list = [vectorizer.transform(corpus_test) for corpus_test in corpus_test_list]
    feature_names = vectorizer.get_feature_names()

    ch2 = SelectKBest(chi2, k=1000)
    x_train = ch2.fit_transform(x_train, y_train)
    x_test_list = [ch2.transform(x_test) for x_test in x_test_list]
    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    x_train = pd.DataFrame(x_train.toarray(), columns=feature_names)
    x_test_list = [pd.DataFrame(x_test.toarray(), columns=feature_names) for x_test in x_test_list]
    return x_train, x_test_list

def preprocess(dataset, X_train, y_train, X_test_list, y_test_list):
    n_test_files = len(X_test_list)

    if "drop_variables" in dataset.keys():
        X_train.drop(columns=dataset['drop_variables'], inplace=True)
        for i in range(n_test_files):
            X_test_list[i].drop(columns=dataset['drop_variables'], inplace=True)

    if "class_imbalance" in dataset.keys() and dataset["class_imbalance"]:
        X_train, y_train = down_sample(X_train, y_train)

    n_tr = X_train.shape[0]
    n_te_list = [X_test.shape[0] for X_test in X_test_list]
    test_split = np.cumsum(n_te_list)[:-1]

    y = pd.concat([y_train, *y_test_list], axis=0)

    if dataset['ml_task'] == 'classification':
        le = LabelEncoder()
        y = le.fit_transform(y.values)
    y_train = y[:n_tr]
    y_test_list = np.split(y[n_tr:], test_split)

    if "text_variables" in dataset.keys():
        text_columns = dataset["text_variables"]
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


    X = pd.concat([X_train, *X_test_list], axis=0)
    X = X[0:100]
    X = pd.get_dummies(X, drop_first=True)
    print(X.columns.tolist())
    print(X.shape)
    sys.exit()
    
    X_train = X[:n_tr, :]
    X_test_list = np.split(X[n_tr:], test_split)
    return X_train, y_train, X_test_list, y_test_list

def down_sample(X, y):
    rus = RandomUnderSampler()
    X_rus, y_rus = rus.fit_sample(X, y)
    indices = rus.sample_indices_
    X_train = X.iloc[indices, :]
    y_train = y.iloc[indices]
    return X_train, y_train

def load_data(dataset, train_dir, test_dir_list):
    train = utils.load_df(dataset, train_dir)
    test_list = [utils.load_df(dataset, test_dir) for test_dir in test_dir_list]
    label = dataset['label']
    features = [v for v in train.columns if not v == label]

    X_train, y_train = train.loc[:, features], train.loc[:, label]
    X_test_list = [test.loc[:, features] for test in test_list]
    y_test_list = [test.loc[:, label] for test in test_list]
    X_train, y_train, X_test_list, y_test_list = preprocess(dataset, X_train, y_train, X_test_list, y_test_list)
    return X_train, y_train, X_test_list, y_test_list  

def parse_searcher(searcher):
    train_accs = searcher.cv_results_['mean_train_score']
    val_accs = searcher.cv_results_['mean_test_score']
    best_idx = searcher.best_index_ 
    best_params = searcher.best_params_
    train_acc, val_acc = train_accs[best_idx], val_accs[best_idx]
    best_model = searcher.best_estimator_
    return best_model, best_params, train_acc, val_acc

def train(dataset_name, error_type, train_file, estimator, params_dist, n_jobs=1, special=False):
    if special:
        X_train, y_train, X_test_list, y_test_list = load_imdb(dataset_name, error_type, train_file)
        test_files = ['dirty', 'clean']
    else:
        dataset = utils.get_dataset(dataset_name)
        train_dir = utils.get_dir(dataset, error_type, train_file + "_train.csv")
        test_files = utils.get_test_files(error_type, train_file)
        test_dir_list = [utils.get_dir(dataset, error_type, test_file + "_test.csv") for test_file in test_files]
        X_train, y_train, X_test_list, y_test_list = load_data(dataset, train_dir, test_dir_list)

    if params_dist is not None:
        searcher = RandomizedSearchCV(estimator, params_dist, cv=5, n_iter=20, n_jobs=n_jobs, return_train_score=True)
        searcher.fit(X_train, y_train)
        best_model, best_params, train_acc, val_acc = parse_searcher(searcher)
    else:
        best_model = estimator
        best_model.fit(X_train, y_train)
        train_acc = best_model.score(X_train, y_train)
        best_params = {}
        val_acc = np.nan

    result_dict = {"best_params": best_params, "train_acc":train_acc, "val_acc": val_acc}

    for X_test, y_test, file in zip(X_test_list, y_test_list, test_files):
        test_acc = best_model.score(X_test, y_test)
        result_dict[file + "_test_acc"] = test_acc        
    return result_dict

# Special Case for IMDB
def load_imdb(dataset_name, error_type, train_file):
    dataset = utils.get_dataset(dataset_name)
    file_dir = utils.get_dir(dataset, error_type)
    file_predix= utils.get_dir(dataset, error_type, train_file)
    X_train = pickle.load(open(file_predix + '_X_train.p', 'rb'), encoding='latin1').toarray()
    y_train = pickle.load(open(file_predix + '_y_train.p', 'rb'), encoding='latin1')
    clean_X_test = pickle.load(open(file_dir + '/clean_X_test.p', 'rb'), encoding='latin1').toarray()
    clean_y_test = pickle.load(open(file_dir + '/clean_y_test.p', 'rb'), encoding='latin1')
    dirty_X_test = pickle.load(open(file_dir + '/dirty_X_test.p', 'rb'), encoding='latin1').toarray()
    dirty_y_test = pickle.load(open(file_dir + '/dirty_y_test.p', 'rb'), encoding='latin1')
    return X_train, y_train, [dirty_X_test, clean_X_test], [dirty_y_test, clean_y_test]





