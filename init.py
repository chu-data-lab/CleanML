""" Initialize datasets:
        Delete missing labels of raw.csv for each dataset to ensure Supervised Learning.
        Delete missing features for dataset not having "missing_values" in config
        Split dataset into train/test
"""
import config
import utils
import pandas as pd
import numpy as np
import os
import argparse

def delete_missing_labels(raw, label_name):
    """ Delete missing labels"""
    label = raw[label_name]
    is_missing_label = label.isnull()
    dirty = raw[is_missing_label == False]
    return dirty

def delete_missing_values(raw):
    """ Delete missing values"""
    dirty = raw.dropna()
    return dirty

def split(data, test_ratio, seed, max_size=None):
    """ Shuffle and split data to train / test"""
    # random shuffle 
    np.random.seed(seed)
    N = data.shape[0]
    idx = np.random.permutation(N)

    # only use first max_size data if N > max_size
    if max_size is not None:
        N = min(N, int(max_size))

    # split train and test
    test_size = int(N * test_ratio)
    idx_train = idx[test_size:N]
    idx_test = idx[:test_size]
    train = data.iloc[idx_train]
    test = data.iloc[idx_test]
    idx_train = pd.DataFrame(idx_train, columns=["index"])
    idx_test = pd.DataFrame(idx_test, columns=["index"])
    return train, test, idx_train, idx_test

def reset(dataset):
    """ Reset dataset"""
    # delete folders for each error
    for error in dataset['error_types']:
        utils.remove(utils.get_dir(dataset, error))
    
    # delete dirty_train and dirty_test in raw folder
    utils.remove(utils.get_dir(dataset, 'raw', 'dirty_train.csv'))
    utils.remove(utils.get_dir(dataset, 'raw', 'dirty_test.csv'))
    utils.remove(utils.get_dir(dataset, 'raw', 'dirty.csv'))
    utils.remove(utils.get_dir(dataset, 'raw', 'idx_train.csv'))
    utils.remove(utils.get_dir(dataset, 'raw', 'idx_test.csv'))
    utils.remove(utils.get_dir(dataset, 'raw', 'version.json'))

def init(dataset, test_ratio=0.3, seed=1, max_size=None):
    """ Initialize dataset: raw -> dirty -> dirty_train, dirty_test
        
        Args:
            dataset (dict): dataset dict in config.py
            max_size (int): maximum limit of dataset size
            test_ratio: ratio of test set
            seed: seed used to split dataset
    """
    print("Initialize dataset {}".format(dataset['data_dir']))

    # load raw data
    raw_path = utils.get_dir(dataset, 'raw', 'raw.csv')
    raw = pd.read_csv(raw_path)

    # delete missing labels or all missing values
    if 'missing_values' not in dataset['error_types']:
        dirty = delete_missing_values(raw)
    else:
        dirty = delete_missing_labels(raw, dataset['label'])

    # split dataset
    train, test, idx_train, idx_test = split(dirty, test_ratio, seed, max_size)

    # save train / test
    save_path_pfx = utils.get_dir(dataset, 'raw', 'dirty')
    utils.save_dfs(train, test, save_path_pfx)

    # save the version (seed) of dataset
    utils.save_version(save_path_pfx, seed)
    
    # save index
    save_path_pfx = utils.get_dir(dataset, 'raw', 'idx')
    utils.save_dfs(idx_train, idx_test, save_path_pfx)

    # save dirty
    # save_path = utils.get_dir(dataset, 'raw', 'dirty.csv')
    # dirty.to_csv(save_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--max_size', type=int, default=None)
    parser.add_argument('--test_ratio', type=int, default=0.3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--reset', default=False, action='store_true' )
    args = parser.parse_args()

    # datasets to be initialized, initialze all datasets if not specified
    datasets = [utils.get_dataset(args.dataset)] if args.dataset is not None else config.datasets
    
    # raw -> dirty
    for dataset in datasets:
        if args.reset:
            reset(dataset)
        else:
            init(dataset, max_size=args.max_size, test_ratio=args.test_ratio, seed=args.seed)