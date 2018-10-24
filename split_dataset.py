import config
import utils
import pandas as pd
import numpy as np
import os

root_dir = config.root_dir
datasets = config.datasets
np.random.seed(1)

def split(dataset, val_ratio=0.2, test_ratio=0.2):
    folder_dir = utils.get_dir(dataset, 'raw')
    file_dir = os.path.join(folder_dir, 'dirty.csv')
    data = pd.read_csv(file_dir)
    data = data.reindex(np.random.permutation(data.index))
    N, m = data.shape
    val_size = int(N * val_ratio)
    test_size = int(N * test_ratio)
    val = data.iloc[:val_size, :]
    test = data.iloc[val_size:val_size + test_size, :]
    train = data.iloc[val_size + test_size:, :]
    train.to_csv(os.path.join(folder_dir, 'dirty_train.csv'), index=False)
    val.to_csv(os.path.join(folder_dir, 'dirty_val.csv'), index=False)
    test.to_csv(os.path.join(folder_dir, 'dirty_test.csv'), index=False)
    return train, val, test

for dataset in datasets:
    print("Split dataset {}".format(dataset['data_dir']))
    train, val, test = split(dataset)
    print(train.shape, val.shape, test.shape)