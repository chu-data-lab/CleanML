from cleaners.MVCleaner import MVCleaner
from cleaners.OutlierCleaner import OutlierCleaner
from cleaners.DuplicatesCleaner import DuplicatesCleaner
from cleaners.InconsistencyCleaner import InconsistencyCleaner
import pandas as pd
import config
import os
import argparse
import sys
import matplotlib.pyplot as plt
import utils
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--mv', default=False, action='store_true')
parser.add_argument('--out', default=False, action='store_true')
parser.add_argument('--dup', default=False, action='store_true')
parser.add_argument('--incon', default=False, action='store_true')
parser.add_argument('--summary', default=False, action='store_true')
parser.add_argument('--dataset', default=None)

def clean_mv(dataset):
    file_dir_pfx = utils.get_dir(dataset, 'raw', 'dirty')
    save_dir = utils.get_dir(dataset, 'missing_values', create_folder=True)
    dirty_train, dirty_test = utils.load_dfs(dataset, file_dir_pfx)

    num_methods = ['mean', 'median', 'mode']
    cat_methods = ['mode', 'dummy']
    cleaners = [MVCleaner(method='impute', num=num, cat=cat) for num in num_methods for cat in cat_methods]
    cleaners.append(MVCleaner(method="delete"))

    for cleaner in cleaners:
        cleaner.fit(dirty_train)
        clean_train, ind_train = cleaner.clean(dirty_train)
        clean_test, ind_test = cleaner.clean(dirty_test)

        clean_dir_pfx = os.path.join(save_dir, 'clean_{}'.format(cleaner.tag))
        if cleaner.method == "delete":
            clean_dir_pfx = os.path.join(save_dir, 'dirty')

        utils.save_dfs(clean_train, clean_test, clean_dir_pfx)
        print('{} finished.'.format(cleaner.tag))

    ind_dir_pfx = os.path.join(save_dir, 'indicator')
    utils.save_dfs(ind_train, ind_test, ind_dir_pfx)

def clean_outliers(dataset):
    if 'missing_values' in dataset['error_types']:
        file_dir_pfx = utils.get_dir(dataset, 'missing_values', 'dirty')
    else:
        file_dir_pfx = utils.get_dir(dataset, 'raw', 'dirty')
    save_dir = utils.get_dir(dataset, 'outliers', create_folder=True)
    dirty_train, dirty_test = utils.load_dfs(dataset, file_dir_pfx)

    detect_methods = ["SD", "IQR", "iso_forest"]
    num_methods = ['mean', 'median', 'mode']
    cat_methods = ['dummy']
    repairers = [MVCleaner(method='impute', num=num, cat=cat) for num in num_methods for cat in cat_methods]
    repairers.append(MVCleaner(method="delete"))

    for detect in detect_methods:
        for repairer in repairers:
            cleaner = OutlierCleaner(detect=detect, repairer=repairer)
            cleaner.fit(dirty_train)
            clean_train, ind_train = cleaner.clean(dirty_train, verbose=True)
            clean_test, ind_test = cleaner.clean(dirty_test, verbose=True, ignore=dataset['label'])

            clean_dir_pfx = os.path.join(save_dir, 'clean_{}'.format(cleaner.tag))
            utils.save_dfs(clean_train, clean_test, clean_dir_pfx)
            ind_dir_pfx = os.path.join(save_dir, 'indicator_{}'.format(detect))
            utils.save_dfs(ind_train, ind_test, ind_dir_pfx)
            print('{} finished.'.format(cleaner.tag))

    dirty_dir_pfx = os.path.join(save_dir, 'dirty')
    utils.save_dfs(dirty_train, dirty_test, dirty_dir_pfx)

def clean_duplicates(dataset):
    if 'inconsistency' in dataset['error_types']:
        file_dir_pfx = utils.get_dir(dataset, "inconsistency", 'clean')
    elif 'missing_values' in dataset['error_types']:
        file_dir_pfx = utils.get_dir(dataset, "missing_values", 'dirty')
    else:
        file_dir_pfx = utils.get_dir(dataset, 'raw', 'dirty')

    save_dir = utils.get_dir(dataset, 'duplicates', create_folder=True)
    dirty_train, dirty_test = utils.load_dfs(dataset, file_dir_pfx)

    cleaner = DuplicatesCleaner()
    clean_train, ind_train = cleaner.clean(dirty_train, dataset['key_columns'])
    clean_test, ind_test = cleaner.clean(dirty_test, dataset['key_columns'])

    clean_dir_pfx = os.path.join(save_dir, 'clean')
    utils.save_dfs(clean_train, clean_test, clean_dir_pfx)
    ind_dir_pfx = os.path.join(save_dir, 'indicator')
    utils.save_dfs(ind_train, ind_test, ind_dir_pfx)
    dirty_dir_pfx = os.path.join(save_dir, 'dirty')
    utils.save_dfs(dirty_train, dirty_test, dirty_dir_pfx)
    print('Finished')

def clean_inconsistency(dataset):
    dirty_dir_pfx = utils.get_dir(dataset, "inconsistency", 'dirty')
    dirty_train, dirty_test = utils.load_dfs(dataset, dirty_dir_pfx)
    clean_train_dir = utils.get_dir(dataset, "inconsistency", 'clean_train.csv')
    clean_train = utils.load_df(dataset, clean_train_dir)

    cleaner = InconsistencyCleaner()
    cleaner.fit(dirty_train, clean_train)
    clean_train, ind_train = cleaner.clean(dirty_train)
    clean_test, ind_test = cleaner.clean(dirty_test)

    clean_dir_pfx = utils.get_dir(dataset, "inconsistency", 'clean')
    ind_dir_pfx = utils.get_dir(dataset, "inconsistency", 'indicator')
    utils.save_dfs(clean_train, clean_test, clean_dir_pfx)
    utils.save_dfs(ind_train, ind_test, ind_dir_pfx)
    print('Finished')

if __name__ == '__main__':
    root_dir = config.root_dir
    datasets = config.datasets
    args = parser.parse_args()
    names = [dt['data_dir'] for dt in datasets]

    if args.dataset is None:
        selected_datasets = datasets
    else:
        if args.dataset not in names:
            print('Dataset does not exist.')
            sys.exit(1)
        selected_datasets = [utils.get_dataset(args.dataset)]

    for dataset in selected_datasets:
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