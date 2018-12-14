import utils
import re
import os
import pandas as pd
import numpy as np
import config
from sklearn.feature_selection import chi2
import sys

def get_indicator_dirs(dataset, error_type):
    save_dir = utils.get_dir(dataset, error_type)
    train_re = re.compile(r'indicator.*train')
    test_re = re.compile(r'indicator.*test')
    if os.path.exists(save_dir):
        train_files = filter(train_re.search, os.listdir(save_dir))
        train_ind_dirs = [utils.get_dir(dataset, error_type, file) for file in train_files]
        test_files = filter(test_re.search, os.listdir(save_dir))
        test_ind_dirs = sorted([utils.get_dir(dataset, error_type, file) for file in test_files])
    else:
        print(save_dir, "not exists")
        sys.exit()
    return train_ind_dirs, test_ind_dirs

def count_cols(ind):
    # Count number of dirty data for each column
    num_dirty = ind.sum()
    num_dirty = num_dirty[num_dirty>0]
    percent_dirty = (num_dirty/len(ind)*100)
    col_names = num_dirty.index
    col_dirty = ["{}({:.2f}%)".format(num_dirty.loc[name], percent_dirty.loc[name]) for name in col_names]
    col_dirty = pd.DataFrame(col_dirty, columns=['# dirty records'], index=col_names)
    count = {"number_dirty_columns":col_dirty}
    return count

def count_rows(ind):
    # Count number of dirty examples
    num_dirty = sum(ind.any(axis=1))
    percent_dirty = (num_dirty/len(ind))
    row_dirty = "{}({:.2%})".format(num_dirty, percent_dirty)
    # Count average number of dirty data for each row
    avg_dirty = "{:.2}".format(sum(ind.sum())/num_dirty)
    count = {"number_dirty_rows":row_dirty, "average_dirty_per_row":avg_dirty}
    return count

def load_label(dataset, file_dir):
    train = utils.load_df(dataset, file_dir)
    label = dataset['label']
    y = train.loc[:, label].values.reshape(-1,1)
    return y

def compute_chi2(ind_train, y):
    # Load label
    is_dirty = ind_train.any(axis=1).values.reshape(-1,1)
    ch_value, p_value = chi2(is_dirty, y)
    return ch_value[0], p_value[0]

def class_dist(ind_train, y):
    is_dirty = ind_train.any(axis=1).values.reshape(-1,1)
    dirty_label = y[is_dirty]
    classes = sorted(np.unique(y))
    dist = [np.sum(dirty_label == c) for c in classes]
    return dist

def postprocess_mv(dataset):
    result = {}
    # Load indicators
    train_ind_dirs, test_ind_dirs = get_indicator_dirs(dataset, 'missing_values')
    ind_train = pd.read_csv(train_ind_dirs[0])
    ind_test = pd.read_csv(test_ind_dirs[0])

    # Dataset size
    N_train = ind_train.shape[0]
    N_test = ind_test.shape[0]

    # Missing columns stats
    # print(count_cols(ind_train))
    # print(count_rows(ind_train))

    # Chi2
    file_dir = utils.get_dir(dataset, 'raw', 'dirty_train.csv')
    label = load_label(dataset, file_dir)
    result['ch_value'], result['p_value'] = compute_chi2(ind_train, label)

    # Class dist
    dist = class_dist(ind_train, label)
    result['class_dist'] = '/'.join([str(d) for d in dist])
    return result

def postprocess_out(dataset):
    train_ind_dirs, test_ind_dirs = get_indicator_dirs(dataset, 'outliers')
    ind_trains = [pd.read_csv(ind_dir) for ind_dir in train_ind_dirs]
    ind_tests = [pd.read_csv(ind_dir) for ind_dir in test_ind_dirs]
    method = [ind_dir.split('/')[-1][10:-10] for ind_dir in train_ind_dirs]

    file_dir = utils.get_dir(dataset, 'outliers', 'dirty_train.csv')
    label = load_label(dataset, file_dir)

    result = {}
    for m, ind_train in zip(method, ind_trains):
        res = {}
        res['ch_value'], res['p_value'] = compute_chi2(ind_train, label)

        # Class dist
        dist = class_dist(ind_train, label)
        res['class_dist'] = '/'.join([str(d) for d in dist])
        result[m] = res
    return result

def postprocess_dup(dataset):
    result = {}
    train_ind_dirs, test_ind_dirs = get_indicator_dirs(dataset, 'duplicates')
    ind_train = pd.read_csv(train_ind_dirs[0])
    ind_test = pd.read_csv(test_ind_dirs[0])

    file_dir = utils.get_dir(dataset, 'duplicates', 'dirty_train.csv')
    label = load_label(dataset, file_dir)
    result['ch_value'], result['p_value'] = compute_chi2(ind_train, label)
    
    # Class dist
    dist = class_dist(ind_train, label)
    result['class_dist'] = '/'.join([str(d) for d in dist])
    return result

def postprocess_incon(dataset):
    result = {}
    train_ind_dirs, test_ind_dirs = get_indicator_dirs(dataset, 'inconsistency')
    ind_train = pd.read_csv(train_ind_dirs[0])
    ind_test = pd.read_csv(test_ind_dirs[0])

    file_dir = utils.get_dir(dataset, 'inconsistency', 'dirty_train.csv')
    label = load_label(dataset, file_dir)
    result['ch_value'], result['p_value'] = compute_chi2(ind_train, label)

    # Class dist
    dist = class_dist(ind_train, label)
    result['class_dist'] = '/'.join([str(d) for d in dist])
    return result

datasets = config.datasets
chi_mv = []
chi_out = []
chi_dup = []
chi_incon = []

for dataset in datasets:
    if 'missing_values' in dataset['error_types']:
        result = postprocess_mv(dataset)
        chi_mv.append([dataset['data_dir'], result['ch_value'], result['p_value'], result['class_dist']])

    if 'outliers' in dataset['error_types']:
        result = postprocess_out(dataset)
        res_list = [dataset['data_dir']]
        for m in ["IQR", "SD", "iso_forest"]:
            res_list +=  [result[m]['ch_value'], result[m]['p_value'], result[m]['class_dist']]
        chi_out.append(res_list)

    if 'duplicates' in dataset['error_types']:
        result = postprocess_dup(dataset)
        chi_dup.append([dataset['data_dir'], result['ch_value'], result['p_value'], result['class_dist']])

    if 'inconsistency' in dataset['error_types']:
        result = postprocess_incon(dataset)
        chi_incon.append([dataset['data_dir'], result['ch_value'], result['p_value'], result['class_dist']]) 



chi_mv = pd.DataFrame(chi_mv, columns=['dataset', 'chi2', 'pvalue', 'dist'])
chi_dup = pd.DataFrame(chi_dup, columns=['dataset', 'chi2', 'pvalue', 'dist'])
chi_incon = pd.DataFrame(chi_incon, columns=['dataset', 'chi2', 'pvalue', 'dist'])
chi_out = pd.DataFrame(chi_out, columns=['dataset', 'chi2', 'pvalue', 'dist', 'chi2', 'pvalue', 'dist', 'chi2', 'pvalue', 'dist'])


chi_mv.to_csv('./postprocess/chi_mv.csv', index=False)
chi_dup.to_csv('./postprocess/chi_dup.csv', index=False)
chi_incon.to_csv('./postprocess/chi_incon.csv', index=False)
chi_out.to_csv('./postprocess/chi_out.csv', index=False)


