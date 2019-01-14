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

def parse_searcher(searcher):
    """ Get results from gridsearch. 

        Args:
            searcher: GridSearchCV object
    """
    train_accs = searcher.cv_results_['mean_train_score']
    val_accs = searcher.cv_results_['mean_test_score']
    best_idx = searcher.best_index_ 
    best_params = searcher.best_params_
    train_acc, val_acc = train_accs[best_idx], val_accs[best_idx]
    best_model = searcher.best_estimator_
    return best_model, best_params, train_acc, val_acc

def train(X_train, y_train, X_test_list, y_test_list, test_files, estimator, param_grid, random_state=1, n_jobs=1):
    """ Train the model 
        
        Args:
            X_train (pd.DataFrame): features (train)
            y_train (pd.DataFrame): label (train)
            X_test_list (list): list of features (test)
            y_test_list (list): list of label (test)
            test_files (list): list of filenames of test set
            estimator (sklearn.model): model
            param_grid (dict): hyper-parameters to tune
            random_state (int): seed for training
            n_jobs (int): num of threads
    """
    np.random.seed(random_state)
    
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
        y_pred = best_model.predict(X_test)
        if len(set(y_test)) > 2:
            test_f1 = f1_score(y_test, y_pred, average='macro')
        else:
            test_f1 = f1_score(y_test, y_pred)
            result_dict[file + "_test_f1"] = test_f1  
    return result_dict