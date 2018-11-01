import numpy as np
import pandas as pd
import argparse
import config
import utils
import sys
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder

def preprocess(dataset, X, y):
    if "drop_variables" in dataset.keys():
        X.drop(columns=dataset['drop_variables'], inplace=True)

    X = pd.get_dummies(X, drop_first=True).values
    if dataset['ml_task'] == 'classification':
        le = LabelEncoder()
        y = le.fit_transform(y.values)
    return X, y

def load_data(dataset, file_dir_predix):
    train, test = utils.load_dfs(dataset, file_dir_predix)
    n_tr, n_te = train.shape[0], test.shape[0]
    data = pd.concat([train, test], axis=0)
    label = dataset['label']
    features = [v for v in data.columns if not v == label]
    X = data.loc[:, features]
    y = data.loc[:, label]
    X, y = preprocess(dataset, X, y)

    X_train, y_train = X[:n_tr, :], y[:n_tr]
    X_test, y_test = X[n_tr:], y[n_tr:]
    return X_train, y_train, X_test, y_test

def parse_searcher(searcher):
    train_accs = searcher.cv_results_['mean_train_score']
    val_accs = searcher.cv_results_['mean_test_score']
    best_idx = searcher.best_index_ 
    best_params = searcher.best_params_
    train_acc, val_acc = train_accs[best_idx], val_accs[best_idx]
    best_model = searcher.best_estimator_
    return best_model, best_params, train_acc, val_acc

def train(dataset_name, error_type, file_prefix, model_fn, params_dist, n_jobs=1):
    dataset = utils.get_dataset(dataset_name)
    file_dir_predix = utils.get_dir(dataset, error_type, file_prefix)
    X_train, y_train, X_test, y_test = load_data(dataset, file_dir_predix)
    estimator = model_fn()
    searcher = RandomizedSearchCV(estimator, params_dist, cv=5, n_iter=20, n_jobs=n_jobs, return_train_score=True)
    searcher.fit(X_train, y_train)
    best_model, best_params, train_acc, val_acc = parse_searcher(searcher)
    test_acc= best_model.score(X_test, y_test)
    return best_params, train_acc, val_acc, test_acc