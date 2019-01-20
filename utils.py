import pandas as pd
import os
import config
import sys
import json
import numpy as np
from matplotlib import pyplot as plt
import shutil


def get_dataset(name):
    """
    Get dataset dict in config.py given name
    """
    datasets = config.datasets
    dataset = [d for d in datasets if d['data_dir'] == name]
    if len(dataset) == 0:
        print('Dataset {} does not exist.'.format(name))
        sys.exit()
    return dataset[0]

def load_df(dataset, file_path):
    """
    Get pandas data frame and preprocess categorical variables

    Args: 
        dataset: dataset in config.py
        file_path: path of data file
    Return:
        df: pandas data frame
    """
    df = pd.read_csv(file_path)
    if 'categorical_variables' in dataset.keys():
        categories = dataset['categorical_variables']
        for cat in categories:
            df[cat] = df[cat].astype(str).replace('nan', np.nan)
    return df

def load_dfs(dataset, file_path_pfx, return_version=False):
    train_dir = file_path_pfx + '_train.csv'
    test_dir = file_path_pfx + '_test.csv'
    train = load_df(dataset, train_dir)
    test = load_df(dataset, test_dir)
    if return_version:
        version = get_version(file_path_pfx)
        return train, test, version
    else:
        return train, test

def save_dfs(train, test, save_path_pfx, version=None):
    train_save_path = save_path_pfx + '_train.csv'
    test_save_path = save_path_pfx + '_test.csv'
    train.to_csv(train_save_path, index=False)
    test.to_csv(test_save_path, index=False)
    if version is not None:
        save_version(save_path_pfx, version)


def get_dir(dataset, folder=None, file=None, create_folder=False):
    """
    Get directory of data file 

    Args:
        dataset: dataset in config.py
        folder: raw/missing_values/outliers/duplicates/inconsistency/mislabel
        file: file name
        create_folder: whether create folder if not exist
    Return:
        data_dir: directory of the whole dataset
        folder_dir (optional): directory of given folder
        file_dir (optional): directory of given file
    """
    data_dir = os.path.join(config.data_dir, dataset['data_dir'])
    if folder is None:
        return data_dir

    folder_dir = os.path.join(data_dir, folder)
    if create_folder and not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

    if file is None:
        return folder_dir
    
    file_dir = os.path.join(folder_dir, file)
    return file_dir

def save_version(file_path_pfx, seed):
    directory, file = os.path.split(file_path_pfx)
    version_path = os.path.join(directory, "version.json")
    if os.path.exists(version_path):
        version = json.load(open(version_path, 'r'))
    else:
        version = {}
    version[file] = str(seed)
    json.dump(version, open(version_path, 'w'))

def get_version(file_path_pfx):
    directory, file = os.path.split(file_path_pfx)
    version_path = os.path.join(directory, "version.json")
    if os.path.exists(version_path):
        version = json.load(open(version_path, 'r'))
        return int(version[file])
    else:
        return None
    
def load_result():
    result_dir = config.result_dir
    if os.path.exists(result_dir):
        result = json.load(open(result_dir, 'r'))
    else:
        result = {}
    return result

def save_result(key, res):
    result = load_result()
    val_acc = res["val_acc"]

    if key in result.keys():
        best_val_acc = result[key]["val_acc"]
    else:
        best_val_acc = float('-inf')

    if val_acc > best_val_acc or np.isnan(val_acc):
        result[key] = res
    json.dump(result, open('./result.json', 'w'))
    
def get_model(model_name):
    models = config.models
    mod = [m for m in models if m['name'] == model_name ]
    if len(mod) == 0:
        print("Invalid model name: {}".format(model_name))
        sys.exit()
    return mod[0]

def get_train_files(error_type):
    if error_type == 'missing_values':
        filenames = ["delete", 
                    "impute_mean_mode", 
                    "impute_mean_dummy", 
                    "impute_median_mode", 
                    "impute_median_dummy", 
                    "impute_mode_mode", 
                    "impute_mode_dummy"]
    elif error_type == 'outliers':
        filenames = ["dirty", 
                     "clean_SD_delete", 
                     "clean_iso_forest_delete", 
                     "clean_IQR_delete", 
                     "clean_SD_impute_mean_dummy", 
                     "clean_IQR_impute_mean_dummy", 
                     "clean_iso_forest_impute_mean_dummy", 
                     "clean_SD_impute_median_dummy",
                     "clean_IQR_impute_median_dummy", 
                     "clean_iso_forest_impute_median_dummy",
                     "clean_SD_impute_mode_dummy",
                     "clean_IQR_impute_mode_dummy", 
                     "clean_iso_forest_impute_mode_dummy"]
    elif error_type == 'mislabel':
        filenames = ["dirty_uniform", 
                     "dirty_major",
                     "dirty_minor",
                     "clean"]
    else:
        filenames = ["dirty", "clean"]
    return filenames
    
def get_test_files(error_type, train_file):
    if error_type == "missing_values":
        if train_file == "delete":
            return get_train_files(error_type)
        else:
            return ["delete", train_file]

    elif error_type == "mislabel":
        if train_file == "clean":
            return get_train_files(error_type)
        else:
            return ["clean", train_file]

    elif error_type == "outliers":
        if train_file == "dirty":
            return get_train_files(error_type)
        else:
            return ["dirty", train_file]
            
    else:
        return ["dirty", "clean"]

def delete_result(dataset_name):
    result = load_result()
    del_key = []
    for k, v in result.items():
        dataset, error, file, model, seed = k.split('/')
        if dataset == dataset_name:
            del_key.append(k)
    for k in del_key:
        print("delete {}".format(k))
        del result[k]
    json.dump(result, open('./result.json', 'w'))

def dict_to_df(dic, row_keys_idx, col_keys_idx):
    """ Convert dict to data frame
        
        Args:
            dic: result dictionary. Keys are tuples.
            row_keys_idx: index of keys for rows, ordered hierarchicallly
            col_keys_idx: index of keys for columns, ordered hierarchicallly
    """
    col_keys = sorted(set([tuple([k[i] for i in col_keys_idx]) for k in dic.keys()]))[::-1]
    row_keys = sorted(set([tuple([k[i] for i in row_keys_idx]) for k in dic.keys()]))[::-1]
    sheet_idx = [i for i in np.arange(len(list(dic.keys())[0])) if i not in row_keys_idx and i not in col_keys_idx]
    sheet_keys = sorted(set([tuple([k[i] for i in sheet_idx]) for k in dic.keys()]))

    if len(sheet_keys) > 1:
        print("sheet key not the same in the same sheet")
        sys.exit()
    else:
        sheet_key = sheet_keys[0]

    order = col_keys_idx + row_keys_idx + sheet_idx

    index = pd.MultiIndex.from_tuples(row_keys)
    columns = pd.MultiIndex.from_tuples(col_keys)
    data = []

    for r in row_keys:
        row = []
        for c in col_keys:
            disorder_key = c + r + sheet_key
            key = tuple([d for o, d in sorted(zip(order, disorder_key))])
            
            if key in dic.keys():
                row.append(dic[key])
            else:
                row.append(np.nan)
        data.append(row)
    df = pd.DataFrame(data, index=index, columns=columns)
    return df

def dict_to_dfs(dic, row_keys_idx, col_keys_idx, df_idx):
    """ Convert dict to dataframes
    
    Args:
        dic: result dictionary. Keys are tuples.
        row_keys_idx: index of keys for rows, ordered hierarchicallly
        col_keys_idx: index of keys for columns, ordered hierarchicallly
    """
    dfs = {}
    df_keys = sorted(set([k[df_idx] for k in dic.keys()]))
    for k in df_keys:
        filtered_dic = {key:value for key, value in dic.items() if key[df_idx] == k}           
        df = dict_to_df(filtered_dic, row_keys_idx, col_keys_idx)
        dfs[k] = df
    return dfs

def dict_to_xls(dic, row_keys_idx, col_keys_idx, save_dir, sheet_idx=None):
    """ Convert dict to excel
    
    Args:
        dic: result dictionary. Keys are tuples.
        row_keys_idx: index of keys for rows, ordered hierarchicallly
        col_keys_idx: index of keys for columns, ordered hierarchicallly
        sheet_idx: index of keys for sheet
    """
    writer = pd.ExcelWriter(save_dir)

    if sheet_idx is None:
        df = dict_to_df(dic, row_keys_idx, col_keys_idx)
        df.to_excel(writer)
    else:
        dfs = dict_to_dfs(dic, row_keys_idx, col_keys_idx, sheet_idx)
        for k, df in dfs.items():
            df.to_excel(writer, '%s'%k)            
    writer.save()

def replace_result(result1_dir, result2_dir, dataset_name):
    result1 = json.load(open(result1_dir, 'r'))
    result2 = json.load(open(result2_dir, 'r'))
    rep = {k:v for k, v in result2.items() if dataset_name == k.split('/')[0]}
    result = {**result1, **rep}
    json.dump(result, open('./result.json', 'w'))

def remove(path):
    if os.path.isfile(path):
        os.remove(path) 
    elif os.path.isdir(path):
        shutil.rmtree(path)