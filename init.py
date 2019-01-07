""" Initialize datasets:
        Clean missing labels of raw.csv for each dataset to ensure Supervised Learning.
        Clean missing features for dataset not having "missing_values" in config
        Split dataset into trian/test
"""
import config
import utils
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None)
args = parser.parse_args()

def split(dataset, test_ratio=0.3, seed=1):
    """Split dirty dataset to train / test"""
    
    # load dirty data
    raw_dir = utils.get_dir(dataset, 'raw')
    dirty_path = os.path.join(raw_dir, 'dirty.csv')
    data = pd.read_csv(dirty_path)
    
    # random shuffle
    np.random.seed(seed)
    idx = np.random.permutation(data.index)
    data = data.reindex(idx)
    
    # split train / test
    N, m = data.shape
    test_size = int(N * test_ratio)
    test = data.iloc[:test_size, :]
    train = data.iloc[test_size:, :]
    # assert((test.values[0] != data.values[0]).all())

    # save train / test
    save_dir_pfx = os.path.join(raw_dir, 'dirty')
    utils.save_dfs(train, test, save_dir_pfx)

    # save index
    idx = pd.DataFrame(idx, columns=['index'])
    idx_test = idx[:test_size]
    idx_train = idx[test_size:]
    idx_dir_pfx = os.path.join(raw_dir, 'idx')
    utils.save_dfs(idx_train, idx_test, idx_dir_pfx)

    return train, test

if __name__ == '__main__':
    # datasets to be initialized, initialze all datasets if not specified
    datasets = [utils.get_dataset(args.dataset)] if args.dataset is not None else config.datasets
    
    # raw -> dirty
    for dataset in datasets:
        print("Initialize dataset {}".format(dataset['data_dir']))

        # load raw data
        raw_path = utils.get_dir(dataset, 'raw', 'raw.csv')
        if not os.path.exists(raw_path):
            continue
        raw = pd.read_csv(raw_path)
        
        # clean records with missing labels for all datasets
        # clean records with missing features for datasets not used for missing values 
        if 'missing_values' not in dataset['error_types']:
            dirty = raw.dropna()
        else:
            label = raw[dataset['label']]
            is_missing_label = label.isnull()
            dirty = raw[is_missing_label == False]

        # save dirty data
        save_path = utils.get_dir(dataset, 'raw', 'dirty.csv')
        dirty.to_csv(save_path, index=False)

    # dirty -> dirty train / test
    for dataset in datasets:
        print("Split dataset {}".format(dataset['data_dir']))
        train, test = split(dataset)
        print(train.shape, test.shape)