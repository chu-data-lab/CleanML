"""Clean datasets"""

from cleaners import *
import pandas as pd
import config
import os
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--mv', default=False, action='store_true')
parser.add_argument('--out', default=False, action='store_true')
parser.add_argument('--dup', default=False, action='store_true')
parser.add_argument('--incon', default=False, action='store_true')
parser.add_argument('--dataset', default=None)
args = parser.parse_args()

def clean_mv(dataset):
    """clean missing values"""

    # create saving folder
    save_dir = utils.get_dir(dataset, 'missing_values', create_folder=True)

    # load dirty data
    dirty_path_pfx = utils.get_dir(dataset, 'raw', 'dirty')
    dirty_train, dirty_test = utils.load_dfs(dataset, dirty_path_pfx)

    # get cleaners with different cleaning methods
    num_methods = ['mean', 'median', 'mode']
    cat_methods = ['mode', 'dummy']
    cleaners = [MVCleaner(method='impute', num=num, cat=cat) for num in num_methods for cat in cat_methods]
    cleaners.append(MVCleaner(method="delete"))

    for cleaner in cleaners:
        # fit on dirty train and clean both train and test
        cleaner.fit(dirty_train)
        clean_train, ind_train = cleaner.clean(dirty_train)
        clean_test, ind_test = cleaner.clean(dirty_test)

        # save clean train and test data
        clean_path_pfx = os.path.join(save_dir, 'clean_{}'.format(cleaner.tag))
        if cleaner.method == "delete":
            clean_path_pfx = os.path.join(save_dir, 'dirty') # name delete mv as dirty
        utils.save_dfs(clean_train, clean_test, clean_path_pfx)
        print('{} finished.'.format(cleaner.tag))

        # save imputaion values
        if cleaner.method != "delete":
            impute = cleaner.impute
            impute_dir = os.path.join(save_dir, '{}.csv'.format(cleaner.tag))
            impute.to_csv(impute_dir)

    # save indicator
    ind_path_pfx = os.path.join(save_dir, 'indicator')
    utils.save_dfs(ind_train, ind_test, ind_path_pfx)

def clean_outliers(dataset):
    """clean outliers"""

    # create saving folder
    save_dir = utils.get_dir(dataset, 'outliers', create_folder=True)

    # load dirty data 
    if 'missing_values' in dataset['error_types']:
        # if raw dataset has missing values, use dataset with mv deleted in missing value folder 
        dirty_path_pfx = utils.get_dir(dataset, 'missing_values', 'dirty')
    else:
        dirty_path_pfx = utils.get_dir(dataset, 'raw', 'dirty')
    dirty_train, dirty_test = utils.load_dfs(dataset, dirth_path_pfx)

    # save dirty data
    dirty_path_pfx = os.path.join(save_dir, 'dirty')
    utils.save_dfs(dirty_train, dirty_test, dirty_path_pfx)

    # get cleaners with different detect and repair methods
    detect_methods = ["SD", "IQR", "iso_forest"]
    num_methods = ['mean', 'median', 'mode']
    cat_methods = ['dummy']
    repairers = [MVCleaner(method='impute', num=num, cat=cat) for num in num_methods for cat in cat_methods]
    repairers.append(MVCleaner(method="delete"))
    cleaners = [OutlierCleaner(detect=detect, repairer=repairer) for detect in detect_methods for repairer in repairers]

    for cleaner in cleaners:
        # fit on dirty train and clean both train and test
        cleaner.fit(dirty_train)
        clean_train, ind_train = cleaner.clean(dirty_train, verbose=True)
        clean_test, ind_test = cleaner.clean(dirty_test, verbose=True, ignore=dataset['label']) # don't clean label outliers in test set

        # save clean data
        clean_path_pfx = os.path.join(save_dir, 'clean_{}'.format(cleaner.tag))
        utils.save_dfs(clean_train, clean_test, clean_path_pfx)

        # save indicator
        ind_path_pfx = os.path.join(save_dir, 'indicator_{}'.format(detect))
        utils.save_dfs(ind_train, ind_test, ind_path_pfx)
        print('{} finished.'.format(cleaner.tag))

def clean_duplicates(dataset):
    """clean duplicates"""

    # create saving folder
    save_dir = utils.get_dir(dataset, 'duplicates', create_folder=True)

    # load dirty data
    if 'inconsistency' in dataset['error_types']:
        # if dataset has inconsistencies, use clean data in inconsistency folder
        dirty_path_pfx = utils.get_dir(dataset, "inconsistency", 'clean') 
    elif 'missing_values' in dataset['error_types']:
        # if dataset has missing values, use dataset with mv deleted in missing value folder 
        dirty_path_pfx = utils.get_dir(dataset, "missing_values", 'dirty')
    else:
        dirty_path_pfx = utils.get_dir(dataset, 'raw', 'dirty')
    dirty_train, dirty_test = utils.load_dfs(dataset, dirty_path_pfx)

    # save dirty data
    dirty_dir_pfx = os.path.join(save_dir, 'dirty')
    utils.save_dfs(dirty_train, dirty_test, dirty_dir_pfx)

    # get cleaners
    cleaner = DuplicatesCleaner()

    # clean train and test seperately
    clean_train, ind_train = cleaner.clean(dirty_train, dataset['key_columns'])
    clean_test, ind_test = cleaner.clean(dirty_test, dataset['key_columns'])

    # save clean data
    clean_path_pfx = os.path.join(save_dir, 'clean')
    utils.save_dfs(clean_train, clean_test, clean_path_pfx)

    # save indicator
    ind_path_pfx = os.path.join(save_dir, 'indicator')
    utils.save_dfs(ind_train, ind_test, ind_path_pfx)

    print('Finished')

def clean_inconsistency(dataset):
    """clean inconsistencies"""
    # only clean test, must clean train manually before clean test

    # load dirty train, clean train, dirty test
    dirty_path_pfx = utils.get_dir(dataset, "inconsistency", 'dirty')
    dirty_train, dirty_test = utils.load_dfs(dataset, dirty_path_pfx)
    clean_train_dir = utils.get_dir(dataset, "inconsistency", 'clean_train.csv')
    clean_train = utils.load_df(dataset, clean_train_dir)

    # get cleaner
    cleaner = InconsistencyCleaner()

    # clean dirty test
    cleaner.fit(dirty_train, clean_train)
    clean_train, ind_train = cleaner.clean(dirty_train)
    clean_test, ind_test = cleaner.clean(dirty_test)

    # save clean test
    clean_path_pfx = utils.get_dir(dataset, "inconsistency", 'clean')
    utils.save_dfs(clean_train, clean_test, clean_path_pfx)

    # save indicaotr
    ind_path_pfx = utils.get_dir(dataset, "inconsistency", 'indicator')
    utils.save_dfs(ind_train, ind_test, ind_path_pfx)
    print('Finished')

if __name__ == '__main__':
    # datasets to be cleaned, clean all datasets if not specified
    datasets = [utils.get_dataset(args.dataset)] if args.dataset is not None else config.datasets

    # clean datasets
    for dataset in datasets:
        if args.mv and 'missing_values' in dataset['error_types']:
            print("Cleaning missing values for {}.".format(dataset['data_dir']))
            clean_mv(dataset)

        if args.out and 'outliers' in dataset['error_types']:
            print("Cleaning outliers for {}.".format(dataset['data_dir']))
            clean_outliers(dataset)

        if args.dup and 'duplicates' in dataset['error_types'] and not dataset['manual_clean_duplicates']:
            print("Cleaning duplicates for {}.".format(dataset['data_dir']))
            clean_duplicates(dataset)

        if args.incon and 'inconsistency' in dataset['error_types']:
            print("Cleaning inconsistency for {}.".format(dataset['data_dir']))
            clean_inconsistency(dataset)