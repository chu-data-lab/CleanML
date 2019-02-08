 import pandas as pd
import os
import config
import sys
import json
import numpy as np
from matplotlib import pyplot as plt
import shutil

# =============================================================================
# Data related utils
# =============================================================================

def get_dataset(name):
    """Get dataset dict in config.py given name

    Args:
        name (string): dataset name
    """ 
    dataset = [d for d in config.datasets if d['data_dir'] == name]
    if len(dataset) == 0:
        print('Dataset {} does not exist.'.format(name))
        sys.exit()
    return dataset[0]

def get_model(name):
    """Get model dict in config.py given name

    Args:
        name (string): model name
    """
    model = [m for m in config.models if m['name'] == name ]
    if len(model) == 0:
        print("Model {} does not exist.".format(name))
        sys.exit()
    return model[0]

def get_dir(dataset, folder=None, file=None, create_folder=False):
    """Get directory or path given dataset, folder name (optional) and filename (optional)

    Args:
        dataset(dict): dataset dict in config.py
        folder (string): raw/missing_values/outliers/duplicates/inconsistency/mislabel
        file (string): file name
        create_folder (bool): whether create folder if not exist
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

def load_df(dataset, file_path):
    """load data file into pandas dataframe and convert categorical variables to string

    Args: 
        dataset (dict): dataset in config.py
        file_path (string): path of data file
    """
    df = pd.read_csv(file_path)
    if 'categorical_variables' in dataset.keys():
        categories = dataset['categorical_variables']
        for cat in categories:
            df[cat] = df[cat].astype(str).replace('nan', np.nan) 
    return df

def load_dfs(dataset, file_path_pfx, return_version=False):
    """load train and test files into pandas dataframes 

    Args:
        dataset (dict): dataset in config.py
        file_path_pfx (string): prefix of data file
        return_version (bool): whether to return the version (split seed) of data
    """
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
    """Save train and test pandas dataframes in csv file

    Args:
        train (pd.DataFrame): training set
        test (pd.DataFrame): test set
        save_path_pfx (string): prefix of save path
        version (int): version of data (optional)
    """
    train_save_path = save_path_pfx + '_train.csv'
    test_save_path = save_path_pfx + '_test.csv'
    train.to_csv(train_save_path, index=False)
    test.to_csv(test_save_path, index=False)
    if version is not None:
        save_version(save_path_pfx, version)

def save_version(file_path_pfx, seed):
    """Save version of data in json file

    Args:
        file_path_pfx (string): prefix of path of data file 
        seed (int): split seed of data
    """
    directory, file = os.path.split(file_path_pfx)
    version_path = os.path.join(directory, "version.json")
    if os.path.exists(version_path):
        version = json.load(open(version_path, 'r'))
    else:
        version = {}
    version[file] = str(seed)
    json.dump(version, open(version_path, 'w'))

def get_version(file_path_pfx):
    """Get version of data 

    Args:
        file_path_pfx (string): prefix of path of data file 
    """
    directory, file = os.path.split(file_path_pfx)
    version_path = os.path.join(directory, "version.json")
    if os.path.exists(version_path):
        version = json.load(open(version_path, 'r'))
        return int(version[file])
    else:
        return None

def remove(path):
    """Remove file or directory

    Args:
        path (string): path of file or directory
    """
    if os.path.isfile(path):
        os.remove(path) 
    elif os.path.isdir(path):
        shutil.rmtree(path)

# =============================================================================
# Training related utils
# =============================================================================

def get_train_files(error_type):
    """Get training files given error type

    Args:
        error_type (string): missing_values/outliers/mislabel/duplicates/inconsistency
    """

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
    """Get test files given error type and training file
       Each error has two types of files: dirty and clean (delete and impute for missing values)
       Test files for one training file include the test file corresponding to itself and all of test 
       files in another type (e.g. For outliers, test files for dirty_train are dirty_test and all of 
       clean_***_test. Test files for outliers clean_SD_delete_train are clean_SD_delete_test and 
       dirty_test.)

    Args:
        error_type (string): missing_values/outliers/mislabel/duplicates/inconsistency
        train_file (string): training file specified in get_train_files()
    """
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

def check_completed(dataset, split_seed, experiment_seed):
    """Check whether all experiments for the dataset with split_seed have been completed
    
    Args:
        dataset (dict): dataset dict in config.py
        split_seed (int): split seed
        experiment_seed (int): experiment seed
    """

    result = load_result(dataset['data_dir'])
    np.random.seed(experiment_seed)
    seeds = np.random.randint(10000, size=config.n_retrain)
    for error in dataset['error_types']:
        for model in config.models:
            for train_file in get_train_files(error):
                for s in seeds:
                    key = "{}/v{}/{}/{}/{}/{}".format(dataset['data_dir'], split_seed, error, train_file, model['name'], s)
                    if key not in result.keys():
                        return False
    return True

# =============================================================================
# Result related utils
# =============================================================================

def load_result(dataset_name=None):
    """Load result of one dataset or all datasets (if no argument) from json to dict

    Args:
        dataset_name (string): dataset name. If not specified, load results of all datasets.
    """
    if dataset_name is None:
        files = [file for file in os.listdir(config.result_dir) if file.endswith('_result.json')]
        result_path = [os.path.join(config.result_dir, file) for file in files]
    else:
        result_path = [os.path.join(config.result_dir, '{}_result.json'.format(dataset_name))]

    result = {}
    for path in result_path:
        if os.path.exists(path):
            result.update(json.load(open(path, 'r')))
    return result

def save_result(dataset_name, key, res):
    """Save result to json

    Args:
        dataset_name (string): dataset name. 
        key (string): key of result in form: dataset_name/split_seed/error_type/clean_method/model_name/seed
        res (dict): result dict {metric_name: metric result}
    """
    result = load_result(dataset_name)
    result[key] = res
    result_path = os.path.join(config.result_dir, '{}_result.json'.format(dataset_name))
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    json.dump(result, open(result_path, 'w'))

def dict_to_df(dic, row_keys_idx, col_keys_idx):
    """Convert dict to data frame
        
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
        print(sheet_keys)
        print("sheet key must be unique in the same sheet.")
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
    """Convert dict to multiple dataframes saved in one dict
    
    Args:
        dic (dict): result dictionary. Keys are tuples.
        row_keys_idx (int): index of keys for rows, ordered hierarchicallly
        col_keys_idx (int): index of keys for columns, ordered hierarchicallly
        df_idx (int): index of keys for spliting dict to multiple dfs.
    """
    dfs = {}
    df_keys = sorted(set([k[df_idx] for k in dic.keys()]))
    for k in df_keys:
        filtered_dic = {key:value for key, value in dic.items() if key[df_idx] == k}           
        df = dict_to_df(filtered_dic, row_keys_idx, col_keys_idx)
        dfs[k] = df
    return dfs

def dict_to_xls(dic, row_keys_idx, col_keys_idx, save_dir, sheet_idx=None):
    """Convert dict to excel
    
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