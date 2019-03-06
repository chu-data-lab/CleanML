"""Clean datasets"""
import pandas as pd
import config
import os
import argparse
import utils
import sys
import schema.clean_method

def clean_error(dataset, error):
    """ Clean one error in the dataset.
    
    Args:
        dataset (dict): dataset dict in dataset.py
        error (string): error type
    """
    # create saving folder
    save_dir = utils.get_dir(dataset, error, create_folder=True)

    # load dirty data
    dirty_path_pfx = utils.get_dir(dataset, 'raw', 'dirty')
    dirty_train, dirty_test, version = utils.load_dfs(dataset, dirty_path_pfx, return_version=True)

    # delete missing values if error type is not missing values
    if error != 'missing_values':
        dirty_train = dirty_train.dropna().reset_index(drop=True)
        dirty_test = dirty_test.dropna().reset_index(drop=True)

    # save dirty data
    dirty_path_pfx = os.path.join(save_dir, 'dirty')
    utils.save_dfs(dirty_train, dirty_test, dirty_path_pfx, version)

    # clean the error in the dataset with various cleaning methods
    error_type = utils.get_error(error)
    for clean_method, cleaner in error_type['clean_methods'].items():
        print("        - Clean the error with method '{}'".format(clean_method))
        # fit on dirty train and clean both train and test
        cleaner.fit(dataset, dirty_train)
        clean_train, ind_train, clean_test, ind_test = cleaner.clean(dirty_train, dirty_test)
 
        # save clean train and test data
        clean_path_pfx = os.path.join(save_dir, clean_method)
        utils.save_dfs(clean_train, clean_test, clean_path_pfx, version)

        # save indicator
        ind_path_pfx = os.path.join(save_dir, 'indicator_{}'.format(clean_method))
        utils.save_dfs(ind_train, ind_test, ind_path_pfx)

def clean(dataset):
    """ Clean each error in the dataset.
    
    Args:
        dataset (dict): dataset dict in dataset.py
    """
    print("- Clean dataset '{}'".format(dataset['data_dir']))
    for error in dataset['error_types']:
        print("    - Clean error type '{}'".format(error))
        clean_error(dataset, error)
    print("    - Finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None)
    args = parser.parse_args()

    # datasets to be cleaned, clean all datasets if not specified
    datasets = [utils.get_dataset(args.dataset)] if args.dataset is not None else config.datasets

    # clean datasets
    for dataset in datasets:
        clean(dataset)