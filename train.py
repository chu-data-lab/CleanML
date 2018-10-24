import numpy as np
import pandas as pd
import argparse
import config
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import utils
import sys

def train_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)
    return train_acc, val_acc, test_acc

def tune_hyper(models, X_train, y_train, X_val, y_val, X_test, y_test):
    train_accs = []
    val_accs = []
    best_acc = 0
    best_model = None

    for model in models:
        train_acc, val_acc, _ = \
            train_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model

    test_acc = best_model.score(X_test, y_test)
    return best_model, best_acc, train_accs, val_accs, test_acc

def load_data(dataset, error_type, file):
    data_dir = utils.get_dir(dataset, error_type, file)
    data = utils.get_df(dataset, data_dir)
    label = [dataset['label']]
    features = [v for v in data.columns if not v == dataset['label']]
    X = pd.DataFrame(data, columns=features)
    y = pd.DataFrame(data, columns=label)
    X = pd.get_dummies(X, drop_first=True)
    y = pd.get_dummies(y, drop_first=True)
    return X.values, y.values

def split_dataset(X, y, val_size=0.2, test_size=0.2, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=random_state)
    return X_train, y_train, X_val, y_val, X_test, y_test

def experiment(dataset, error_type, file, model):
    X, y= load_data(dataset, error_type, file)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
    train_acc, val_acc, test_acc = \
        train_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test)
    return train_acc, val_acc, test_acc

if __name__ == '__main__':
    dataset = utils.get_dataset('Airbnb')
    model = LinearRegression() 
    filenames = ['clean_mv_delete.csv', 'clean_mv_impute_mean_dummy.csv', 'clean_mv_impute_mean_mode.csv', 
    'clean_mv_impute_median_dummy.csv', 'clean_mv_impute_median_mode.csv', 'clean_mv_impute_mode_dummy.csv', 'clean_mv_impute_mode_mode.csv']

    for file in filenames:
        train_acc, val_acc, test_acc = experiment(dataset, 'missing_values', file, model)
        print(train_acc, val_acc)
    










