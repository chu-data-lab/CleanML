"""Clean datasets"""
from cleaners import *
import pandas as pd
import config
import os
import argparse
import utils
from inject import inject
import sys

def clean_mv(dataset):
    """Clean missing values
    
    Args:
        dataset (dict): dataset dict in config
    """

    # create saving folder
    save_dir = utils.get_dir(dataset, 'missing_values', create_folder=True)

    # load dirty data
    dirty_path_pfx = utils.get_dir(dataset, 'raw', 'dirty')
    dirty_train, dirty_test, version = utils.load_dfs(dataset, dirty_path_pfx, return_version=True)

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
        clean_path_pfx = os.path.join(save_dir, cleaner.tag)
        utils.save_dfs(clean_train, clean_test, clean_path_pfx, version)
        # print('{} finished.'.format(cleaner.tag))

        # save imputaion values
        if cleaner.method != "delete":
            impute = cleaner.impute
            impute_dir = os.path.join(save_dir, '{}_imputation_value.csv'.format(cleaner.tag))
            impute.to_csv(impute_dir)

    # save indicator
    ind_path_pfx = os.path.join(save_dir, 'indicator')
    utils.save_dfs(ind_train, ind_test, ind_path_pfx)

def clean_outliers(dataset):
    """Clean outliers
    
    Args:
        dataset (dict): dataset dict in config
    """

    # create saving folder
    save_dir = utils.get_dir(dataset, 'outliers', create_folder=True)

    # load dirty data 
    if 'missing_values' in dataset['error_types']:
        # if raw dataset has missing values, use dataset with mv deleted in missing value folder 
        dirty_path_pfx = utils.get_dir(dataset, 'missing_values', 'delete')
    else:
        dirty_path_pfx = utils.get_dir(dataset, 'raw', 'dirty')
    dirty_train, dirty_test, version = utils.load_dfs(dataset, dirty_path_pfx, return_version=True)

    # save dirty data
    dirty_path_pfx = os.path.join(save_dir, 'dirty')
    utils.save_dfs(dirty_train, dirty_test, dirty_path_pfx, version)

    # get cleaners with different detect and repair methods
    detect_methods = ["SD", "IQR", "iso_forest"]
    num_methods = ['mean', 'median', 'mode']
    cat_methods = ['dummy']
    repairers = [MVCleaner(method='impute', num=num, cat=cat) for num in num_methods for cat in cat_methods]
    repairers.append(MVCleaner(method="delete"))
    cleaners = [OutlierCleaner(detect_method=detect, repairer=repairer) for detect in detect_methods for repairer in repairers]

    for cleaner in cleaners:
        # fit on dirty train and clean both train and test
        cleaner.fit(dirty_train)
        clean_train, ind_train = cleaner.clean(dirty_train)
        clean_test, ind_test = cleaner.clean(dirty_test, ignore=dataset['label']) # don't clean label outliers in test set

        # save clean data
        clean_path_pfx = os.path.join(save_dir, 'clean_{}'.format(cleaner.tag))
        utils.save_dfs(clean_train, clean_test, clean_path_pfx, version)

        # save indicator
        ind_path_pfx = os.path.join(save_dir, 'indicator_{}'.format(cleaner.detect_method))
        utils.save_dfs(ind_train, ind_test, ind_path_pfx)
        # print('{} finished.'.format(cleaner.tag))

def clean_duplicates(dataset):
    """Clean duplicates
    
    Args:
        dataset (dict): dataset dict in config
    """

    # create saving folder
    save_dir = utils.get_dir(dataset, 'duplicates', create_folder=True)

    # load dirty data
    if 'inconsistency' in dataset['error_types']:
        # if dataset has inconsistencies, use clean data in inconsistency folder
        dirty_path_pfx = utils.get_dir(dataset, "inconsistency", 'clean') 
    elif 'missing_values' in dataset['error_types']:
        # if dataset has missing values, use dataset with mv deleted in missing value folder 
        dirty_path_pfx = utils.get_dir(dataset, "missing_values", 'delete')
    else:
        dirty_path_pfx = utils.get_dir(dataset, 'raw', 'dirty')
    dirty_train, dirty_test, version = utils.load_dfs(dataset, dirty_path_pfx, return_version=True)

    # save dirty data
    dirty_path_pfx = os.path.join(save_dir, 'dirty')
    utils.save_dfs(dirty_train, dirty_test, dirty_path_pfx, version)

    # get cleaners
    cleaner = DuplicatesCleaner()

    # clean train and test seperately
    clean_train, ind_train = cleaner.clean(dirty_train, dataset['key_columns'])
    clean_test, ind_test = cleaner.clean(dirty_test, dataset['key_columns'])

    # save clean data
    clean_path_pfx = os.path.join(save_dir, 'clean')
    utils.save_dfs(clean_train, clean_test, clean_path_pfx, version)

    # save indicator
    ind_path_pfx = os.path.join(save_dir, 'indicator')
    utils.save_dfs(ind_train, ind_test, ind_path_pfx)

    # print('Finished')

def clean_inconsistency(dataset):
    """Clean inconsistencies
    
    Args:
        dataset (dict): dataset dict in config
    """

    # create saving folder
    save_dir = utils.get_dir(dataset, 'inconsistency', create_folder=True)

    # load dirty data 
    if 'missing_values' in dataset['error_types']:
        # if raw dataset has missing values, use dataset with mv deleted in missing value folder 
        dirty_path_pfx = utils.get_dir(dataset, 'missing_values', 'delete')
    else:
        dirty_path_pfx = utils.get_dir(dataset, 'raw', 'dirty')
    dirty_train, dirty_test, version = utils.load_dfs(dataset, dirty_path_pfx, return_version=True)

    # save dirty data
    dirty_path_pfx = os.path.join(save_dir, 'dirty')
    utils.save_dfs(dirty_train, dirty_test, dirty_path_pfx, version)

    # get cleaner
    cleaner = InconsistencyCleaner()

    # fit cleaner
    dirty_raw_path = utils.get_dir(dataset, 'raw', 'raw.csv')
    clean_raw_path = utils.get_dir(dataset, 'raw', 'inconsistency_clean_raw.csv')
    if not os.path.exists(clean_raw_path):
        print("Must provide clean version of raw data for cleaning inconsistency")
        sys.exit(1)
    dirty_raw = utils.load_df(dataset, dirty_raw_path)
    clean_raw = utils.load_df(dataset, clean_raw_path)

    # clean dirty data
    cleaner.fit(dirty_raw, clean_raw)
    clean_train, ind_train = cleaner.clean(dirty_train)
    clean_test, ind_test = cleaner.clean(dirty_test)

    # save clean data
    clean_path_pfx = utils.get_dir(dataset, "inconsistency", 'clean')
    utils.save_dfs(clean_train, clean_test, clean_path_pfx, version)

    # save indicaotr
    ind_path_pfx = utils.get_dir(dataset, "inconsistency", 'indicator')
    utils.save_dfs(ind_train, ind_test, ind_path_pfx)
    # print('Finished')

def clean(dataset):
    if 'missing_values' in dataset['error_types']:
        print("Clean missing values for {}.".format(dataset['data_dir']))
        clean_mv(dataset)

    if 'outliers' in dataset['error_types']:
        print("Clean outliers for {}.".format(dataset['data_dir']))
        clean_outliers(dataset)

    if 'inconsistency' in dataset['error_types']:
        print("Clean inconsistency for {}.".format(dataset['data_dir']))
        clean_inconsistency(dataset)

    if 'duplicates' in dataset['error_types']:
        print("Clean duplicates for {}.".format(dataset['data_dir']))
        clean_duplicates(dataset)

    if 'mislabel' in dataset['error_types']:
        print("Inject mislabel for {}.".format(dataset['data_dir']))
        inject(dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None)
    args = parser.parse_args()

    # datasets to be cleaned, clean all datasets if not specified
    datasets = [utils.get_dataset(args.dataset)] if args.dataset is not None else config.datasets

    # clean datasets
    for dataset in datasets:
        clean(dataset)