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
from sklearn.model_selection import cross_val_score

def parse_searcher(searcher):
    """Get results from gridsearch

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

def train(X_train, y_train, estimator, param_grid, seed=1, n_jobs=1):
    """Train the model 
        
    Args:
        X_train (pd.DataFrame): features (train)
        y_train (pd.DataFrame): label (train)
        estimator (sklearn.model): model
        param_grid (dict): hyper-parameters to tune
        seed (int): seed for training
        n_jobs (int): num of threads
    """
    np.random.seed(seed)
    
    # train and tune hyper parameter with 5-fold cross validation
    if param_grid is not None:
        searcher = GridSearchCV(estimator, param_grid, cv=5, n_jobs=n_jobs, return_train_score=True, iid=False)
        searcher.fit(X_train, y_train)
        best_model, best_params, train_acc, val_acc = parse_searcher(searcher)
    else: 
        # if no hyper parameter is given, train directly
        best_model = estimator
        val_acc = cross_val_score(best_model, X_train, y_train, cv=5).mean()
        best_model.fit(X_train, y_train)
        train_acc = best_model.score(X_train, y_train)
        best_params = {}

    result = {"best_params": best_params, "train_acc":train_acc, "val_acc": val_acc}
    return best_model, result

def evaluate(best_model, X_test_list, y_test_list, test_files):
    # evaluate on test sets
    result = {}
    for X_test, y_test, file in zip(X_test_list, y_test_list, test_files):
        y_pred = best_model.predict(X_test)
        test_acc = np.mean(y_pred == y_test)
        result[file + "_test_acc"] = test_acc

        if len(set(y_test)) > 2:
            test_f1 = f1_score(y_test, y_pred, average='macro')
        else:
            test_f1 = f1_score(y_test, y_pred)
            result[file + "_test_f1"] = test_f1  
    return result

def get_coarse_grid(model, seed, n_jobs, N):
    """Get hyper parameters (coarse random search) """
    np.random.seed(seed)
    low, high = model["hyperparams_range"]
    if model["hyperparams_type"] == "real":
        param_grid = {model['hyperparams']: 10 ** np.random.uniform(low, high, 20)}
    if model["hyperparams_type"] == "int":
        if model["name"] == "knn_classification":
            high = min(high, int(N/5*4))
        param_grid = {model['hyperparams']: np.random.randint(low, high, 20)}
    return param_grid

def get_fine_grid(model, best_param_coarse, n_jobs, N):
    """Get hyper parameters (fine grid search, around the best parameter in coarse search) """
    if model["hyperparams_type"] == "real":
        base = np.log10(best_param_coarse)
        param_grid = {model['hyperparams']: np.linspace(10**(base-0.5), 10**(base+0.5), 20)}
    if model["hyperparams_type"] == "int":
        low = max(best_param_coarse - 10, 1)
        high = low + 20
        if model["name"] == "knn_classification":
            high = min(high, int(N/5*4))
        param_grid = {model['hyperparams']: np.arange(low, high)}
    return param_grid

def hyperparam_search(X_train, y_train, model, n_jobs=1, seed=1):
    np.random.seed(seed)
    coarse_param_seed, coarse_train_seed, fine_train_seed = np.random.randint(1000, size=3)
    fixed_params = model["fixed_params"]
    if "parallelable" in model.keys() and model['parallelable']:
        fixed_params["n_jobs"] = n_jobs
    estimator = model["fn"](**fixed_params)

    # hyperparameter search
    if "hyperparams" not in model.keys():
        # if no hyper parmeter, train directly
        best_model, result = train(X_train, y_train, estimator, None, n_jobs=n_jobs, seed=coarse_train_seed)
    else:
        # coarse random search
        param_grid = get_coarse_grid(model, coarse_param_seed, n_jobs, len(y_train))
        best_model_coarse, result_coarse = train(X_train, y_train, estimator, param_grid, n_jobs=n_jobs, seed=coarse_train_seed)
        val_acc_coarse = result_coarse['val_acc']
        
        # fine grid search
        best_param_coarse = result_coarse['best_params'][model['hyperparams']]
        param_grid = get_fine_grid(model, best_param_coarse, n_jobs, len(y_train))
        best_model_fine, result_fine = train(X_train, y_train, estimator, param_grid, n_jobs=n_jobs, seed=fine_train_seed)
        val_acc_fine = result_fine['val_acc']

        if val_acc_fine > val_acc_coarse:
            result = result_fine
            best_model = best_model_fine
        else:
            result = result_coarse
            best_model = best_model_coarse

        # convert int to float to avoid json error
        if model["hyperparams_type"] == "int":
            result['best_params'][model["hyperparams"]] *= 1.0

    return best_model, result

def train_and_evaluate(X_train, y_train, X_test_list, y_test_list, test_files, model, n_jobs=1, seed=1):
    """Search hyperparameters and evaluate
        
    Args:
        X_train (pd.DataFrame): features (train)
        y_train (pd.DataFrame): label (train)
        X_test_list (list): list of features (test)
        y_test_list (list): list of label (test)
        test_files (list): list of filenames of test set
        model (dict): ml model dict in model.py
        seed (int): seed for training
        n_jobs (int): num of threads
    """
    best_model, result_train = hyperparam_search(X_train, y_train, model, n_jobs, seed)
    result_test = evaluate(best_model, X_test_list, y_test_list, test_files)
    result = {**result_train, **result_test}
    return result