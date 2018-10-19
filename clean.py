from cleaners.MVCleaner import MVCleaner
from cleaners.OutlierCleaner import OutlierCleaner
from cleaners.DuplicatesCleaner import DuplicatesCleaner
import pandas as pd
import config
import os
import argparse
import sys
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--mv', default=False, action='store_true')
parser.add_argument('--out', default=False, action='store_true')
parser.add_argument('--dup', default=False, action='store_true')
parser.add_argument('--summary', default=False, action='store_true')
parser.add_argument('--dataset', default=None)

def get_dir(dataset, error_type, source='raw'):
    data_dir = os.path.join(root_dir, dataset['data_dir'])
    file_dir = os.path.join(data_dir, 'raw', 'dirty.csv')

    if source == 'delete_mv':
        file_dir = os.path.join(data_dir, 'missing_values', 'clean_mv_delete.csv')
        if not os.path.exists(file_dir):
            print('Must first clean missing values.')
            sys.exit()

    if source == 'clean_incon':
        file_dir = os.path.join(data_dir, 'inconsistency', 'clean.csv')
        if not os.path.exists(file_dir):
            print('Must first clean inconsistency.')
            sys.exit()

    save_dir = os.path.join(data_dir, error_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return data_dir, file_dir, save_dir

def get_df(dataset, file_dir):
    df = pd.read_csv(file_dir)
    categories = dataset['categorical_variables']
    for cat in categories:
        df[cat] = df[cat].astype(str)
    return df

def clean_mv(dataset):
    data_dir, file_dir, save_dir = get_dir(dataset, 'missing_values')
    dirty_data = get_df(dataset, file_dir)

    num_methods = ['mean', 'median', 'mode']
    cat_methods = ['mode', 'dummy']
    cleaners = [MVCleaner(method='impute', num=num, cat=cat) for num in num_methods for cat in cat_methods]
    cleaners.append(MVCleaner(method="delete"))

    for cleaner in cleaners:
        clean_data, mv_mat = cleaner.clean(dirty_data)
        data_save_dir = os.path.join(save_dir, 'clean_mv_{}.csv'.format(cleaner.tag))
        clean_data.to_csv(data_save_dir, index=False)
        print('{} finished.'.format(cleaner.tag))

    mat_save_dir = os.path.join(save_dir, 'indicator.csv')
    dirty_save_dir = os.path.join(save_dir, 'dirty.csv')
    dirty_data.to_csv(dirty_save_dir, index=False)
    mv_mat.to_csv(mat_save_dir, index=False)
    
def clean_outliers(dataset):
    if 'mv' in dataset['error_types']:
        data_dir, file_dir, save_dir = get_dir(dataset, "outliers", 'delete_mv')
    else:
        data_dir, file_dir, save_dir = get_dir(dataset, "outliers")
    
    dirty_data = get_df(dataset, file_dir)
    detect_methods = ["SD", "IQR", "iso_forest"]
    
    num_methods = ['mean', 'median', 'mode']
    cat_methods = ['dummy']
    repairers = [MVCleaner(method='impute', num=num, cat=cat) for num in num_methods for cat in cat_methods]
    repairers.append(MVCleaner(method="delete"))

    for detect in detect_methods:
        detector = OutlierCleaner(detect=detect)
        out_mat = detector.detect(dirty_data, verbose=True)
        for repairer in repairers:
            cleaner = OutlierCleaner(detect=detect, repairer=repairer)
            clean_data = cleaner.repair(dirty_data, out_mat)
            data_save_dir = os.path.join(save_dir, 'clean_{}.csv'.format(cleaner.tag))
            out_mat_save_dir = os.path.join(save_dir, 'indicator_{}.csv'.format(detect))
            clean_data.to_csv(data_save_dir, index=False)
            out_mat.to_csv(out_mat_save_dir, index=False)
            print('{} finished.'.format(cleaner.tag))

    dirty_save_dir = os.path.join(save_dir, 'dirty.csv')
    dirty_data.to_csv(dirty_save_dir, index=False)

def clean_duplicates(dataset):
    if 'incon' in dataset['error_types']:
        data_dir, file_dir, save_dir = get_dir(dataset, "duplicates", 'clean_incon')
    elif 'mv' in dataset['error_types']:
        data_dir, file_dir, save_dir = get_dir(dataset, "duplicates", 'delete_mv')
    else:
        data_dir, file_dir, save_dir = get_dir(dataset, "duplicates")

    dirty_data = get_df(dataset, file_dir)
    cleaner = DuplicatesCleaner()
    clean_data, is_dup = cleaner.clean(dirty_data, dataset['key_columns'])
    
    data_save_dir = os.path.join(save_dir, 'clean.csv')
    is_dup_save_dir = os.path.join(save_dir, 'indicator.csv')
    dirty_save_dir = os.path.join(save_dir, 'dirty.csv')
    clean_data.to_csv(data_save_dir, index=False)
    is_dup.to_csv(is_dup_save_dir, index=False)
    dirty_data.to_csv(dirty_save_dir, index=False)
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
        selected_datasets = [dt for dt in datasets if dt['data_dir'] == args.dataset]

    for dataset in selected_datasets:
        if args.mv and 'mv' in dataset['error_types']:
            print("Cleaning missing values for {}.".format(dataset['data_dir']))
            clean_mv(dataset)

        if args.out and 'out' in dataset['error_types']:
            print("Cleaning outliers for {}.".format(dataset['data_dir']))
            clean_outliers(dataset)

        if args.dup and 'dup' in dataset['error_types'] and not dataset['dup_ground_truth']:
            print("Cleaning duplicates for {}.".format(dataset['data_dir']))
            clean_duplicates(dataset)