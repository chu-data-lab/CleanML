import numpy as np
import pandas as pd
import argparse
import config
import utils
import sys
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pickle
import preprocess
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, chi2

def load_data(dataset, train_dir, test_dir_list, normalize, down_sample_seed=1):
    train = utils.load_df(dataset, train_dir)
    test_list = [utils.load_df(dataset, test_dir) for test_dir in test_dir_list]
    label = dataset['label']
    features = [v for v in train.columns if not v == label]

    X_train, y_train = train.loc[:, features], train.loc[:, label]
    X_test_list = [test.loc[:, features] for test in test_list]
    y_test_list = [test.loc[:, label] for test in test_list]

    X_train, y_train, X_test_list, y_test_list = \
        preprocess.preprocess(dataset, X_train, y_train, X_test_list, y_test_list, normalize, down_sample_seed)
    return X_train, y_train, X_test_list, y_test_list  

def parse_searcher(searcher):
    train_accs = searcher.cv_results_['mean_train_score']
    val_accs = searcher.cv_results_['mean_test_score']
    best_idx = searcher.best_index_ 
    best_params = searcher.best_params_
    train_acc, val_acc = train_accs[best_idx], val_accs[best_idx]
    best_model = searcher.best_estimator_
    return best_model, best_params, train_acc, val_acc

def train(dataset_name, error_type, train_file, estimator, param_gird, random_state=1, n_jobs=1, special=False, normalize=False):
    np.random.seed(random_state)
    dataset = utils.get_dataset(dataset_name)
    if special:
        X_train, y_train, X_test_list, y_test_list = load_imdb(dataset, error_type, train_file)
        test_files = ['dirty', 'clean']
    else:
        train_dir = utils.get_dir(dataset, error_type, train_file + "_train.csv")
        test_files = utils.get_test_files(error_type, train_file)
        test_dir_list = [utils.get_dir(dataset, error_type, test_file + "_test.csv") for test_file in test_files]
        down_sample_seed = np.random.randint(1000)
        X_train, y_train, X_test_list, y_test_list = load_data(dataset, train_dir, test_dir_list, normalize, down_sample_seed)

    if param_gird is not None:
        searcher = GridSearchCV(estimator, param_gird, cv=5, n_jobs=n_jobs, return_train_score=True, iid=False)
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
        if dataset['ml_task'] == "classification":
            y_pred = best_model.predict(X_test)
            if len(set(y_test)) > 2:
                test_f1 = f1_score(y_test, y_pred, average='macro')
            else:
                test_f1 = f1_score(y_test, y_pred)
                result_dict[file + "_test_f1"] = test_f1  
    return result_dict

# Special Case for IMDB
def load_imdb(dataset, error_type, train_file):
    file_dir = utils.get_dir(dataset, error_type)
    file_predix= utils.get_dir(dataset, error_type, train_file)
    X_train_raw = pickle.load(open(file_predix + '_X_train.p', 'rb'), encoding='latin1').toarray()
    y_train = pickle.load(open(file_predix + '_y_train.p', 'rb'), encoding='latin1')

    clean_X_test = pickle.load(open(file_dir + '/clean_X_test.p', 'rb'), encoding='latin1').toarray()
    clean_y_test = pickle.load(open(file_dir + '/clean_y_test.p', 'rb'), encoding='latin1')
    dirty_X_test = pickle.load(open(file_dir + '/dirty_X_test.p', 'rb'), encoding='latin1').toarray()
    dirty_y_test = pickle.load(open(file_dir + '/dirty_y_test.p', 'rb'), encoding='latin1')
    X_test_list_raw = [dirty_X_test, clean_X_test]
    y_test_list = [dirty_y_test, clean_y_test]
    
    print(X_train_raw.shape)
    ch2 = SelectKBest(chi2, k=200)
    X_train = ch2.fit_transform(X_train_raw, y_train)
    print(X_train.shape)
    X_test_list = [ch2.transform(x_test) for x_test in x_test_list_raw]
    return X_train, y_train, X_test_list, y_test_list