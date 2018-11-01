"""
    Clean missing labels of raw.csv for all dataset to ensure Supervised Learning.
    Clean missing features for dataset not having "missing_values" in config
    Split dataset into trian/val/test
"""

import config
import utils
import pandas as pd
import numpy as np
import os

root_dir = config.root_dir
datasets = config.datasets
np.random.seed(1)

def split(dataset, test_ratio=0.3):
    folder_dir = utils.get_dir(dataset, 'raw')
    file_dir = os.path.join(folder_dir, 'dirty.csv')
    data = pd.read_csv(file_dir)
    idx = np.random.permutation(data.index)
    data = data.reindex(idx)
    
    N, m = data.shape
    test_size = int(N * test_ratio)
    test = data.iloc[:test_size, :]
    train = data.iloc[test_size:, :]

    idx = pd.DataFrame(idx, columns=['index'])
    idx_test = idx[:test_size]
    idx_train = idx[test_size:]

    save_dir_predix = os.path.join(folder_dir, 'dirty')
    idx_dir_predix = os.path.join(folder_dir, 'idx')
    utils.save_dfs(train, test, save_dir_predix)
    utils.save_dfs(idx_train, idx_test, idx_dir_predix)
    return train, test

for dataset in datasets:
    print("Initialize dataset {}".format(dataset['data_dir']))
    raw_dir = utils.get_dir(dataset, 'raw', 'raw.csv')
    save_dir = utils.get_dir(dataset, 'raw', 'dirty.csv')
    if os.path.exists(raw_dir):
        raw = pd.read_csv(raw_dir)
        if 'missing_values' not in dataset['error_types']:
            dirty = raw.dropna()
        else:
            label = raw[dataset['label']]
            is_missing_label = label.isnull()
            dirty = raw[is_missing_label == False]
        dirty.to_csv(save_dir, index=False)

for dataset in datasets:
    print("Split dataset {}".format(dataset['data_dir']))
    train, test = split(dataset)
    print(train.shape, test.shape)