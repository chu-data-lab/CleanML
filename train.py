"""Train the model"""

import numpy as np
import pandas as pd
import argparse
import config
import utils
import sys
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pickle
from preprocess import preprocess
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, chi2

def load_data(dataset, train_dir, test_dir_list, normalize=True, down_sample_seed=1):
    """ Load and preprocess data.

        Args: 
            dataset (dict): dataset dict in config
            train_dir (string): path for training data
            test_dir_list (list): a list of paths for test data (missing values and outlier have multiple test sets)
            normalize (bool): whehter to standarize the data
            down_sample_seed (int): seed for downsampling
    """

    # load data 
    train = utils.load_df(dataset, train_dir)
    test_list = [utils.load_df(dataset, test_dir) for test_dir in test_dir_list]
    
    # split X, y
    label = dataset['label']
    features = [v for v in train.columns if not v == label]
    X_train, y_train = train.loc[:, features], train.loc[:, label]
    X_test_list = [test.loc[:, features] for test in test_list]
    y_test_list = [test.loc[:, label] for test in test_list]

    # preprocess
    X_train, y_train, X_test_list, y_test_list = \
        preprocess(dataset, X_train, y_train, X_test_list, y_test_list, normalize, down_sample_seed)
    return X_train, y_train, X_test_list, y_test_list  

def parse_searcher(searcher):
    """ get results from gridsearch. """

    train_accs = searcher.cv_results_['mean_train_score']
    val_accs = searcher.cv_results_['mean_test_score']
    best_idx = searcher.best_index_ 
    best_params = searcher.best_params_
    train_acc, val_acc = train_accs[best_idx], val_accs[best_idx]
    best_model = searcher.best_estimator_
    return best_model, best_params, train_acc, val_acc

def train(dataset, error_type, train_file, estimator, param_grid, random_state=1, n_jobs=1, normalize=False):
    """ Train the model 
        
        Args:
            dataset (dict): dataset dict in config
            error_type (string): error type
            train_file (string): prefix of training file
            estimator (sklearn.model): model
            param_grid (dict): hyper-parameters to tune
            random_state (int): seed for experiment
            n_jobs (int): num of threads
            normalize (bool): whehter to standarize the data
    """

    np.random.seed(random_state)

    # get train file and test files
    train_dir = utils.get_dir(dataset, error_type, train_file + "_train.csv")
    test_files = utils.get_test_files(error_type, train_file)
    test_dir_list = [utils.get_dir(dataset, error_type, test_file + "_test.csv") for test_file in test_files]
    
    # load data and preprocess
    down_sample_seed = np.random.randint(1000)
    X_train, y_train, X_test_list, y_test_list = load_data(dataset, train_dir, test_dir_list, normalize, down_sample_seed)

    # train and tune hyper parameter with 5-fold cross validation
    if param_grid is not None:
        searcher = GridSearchCV(estimator, param_grid, cv=5, n_jobs=n_jobs, return_train_score=True, iid=False)
        searcher.fit(X_train, y_train)
        best_model, best_params, train_acc, val_acc = parse_searcher(searcher)
    else: 
        # if no hyper parameter is given, train directly
        best_model = estimator
        best_model.fit(X_train, y_train)
        train_acc = best_model.score(X_train, y_train)
        best_params = {}
        val_acc = np.nan

    result_dict = {"best_params": best_params, "train_acc":train_acc, "val_acc": val_acc}

    # evaluate on test sets
    for X_test, y_test, file in zip(X_test_list, y_test_list, test_files):
        test_acc = best_model.score(X_test, y_test)
        result_dict[file + "_test_acc"] = test_acc
        if dataset['ml_task'] == "classification":
            y_pred = best_model.predict(X_test)
            if len(set(y_test)) > 2:
                test_f1 = f1_score(y_test, y_pred, average='macro')
            else:
                test_f1 = f1_score(y_test, y_pred)
                result_dict[file + "_test_f1"] = test_f1  
    return result_dict