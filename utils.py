import pandas as pd
import os
import config
import sys
import json
import numpy as np
from matplotlib import pyplot as plt
import shutil
from collections import defaultdict

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

def get_error(name):
    """Get error dict in config.py given name

    Args:
        name (string): dataset name
    """ 
    error_type = [e for e in config.error_types if e['name'] == name]
    if len(error_type) == 0:
        print('Error type {} does not exist.'.format(name))
        sys.exit()
    return error_type[0]

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
                     "clean_IF_delete", 
                     "clean_IQR_delete", 
                     "clean_SD_impute_mean_dummy", 
                     "clean_IQR_impute_mean_dummy", 
                     "clean_IF_impute_mean_dummy", 
                     "clean_SD_impute_median_dummy",
                     "clean_IQR_impute_median_dummy", 
                     "clean_IF_impute_median_dummy",
                     "clean_SD_impute_mode_dummy",
                     "clean_IQR_impute_mode_dummy", 
                     "clean_IF_impute_mode_dummy"]
    elif error_type == 'mislabel':
        filenames = ["dirty",
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

def load_result(dataset_name=None, parse_key=False):
    """Load result of one dataset or all datasets (if no argument) from json to dict

    Args:
        dataset_name (string): dataset name. If not specified, load results of all datasets.
        parse_key (bool): whether convert key from string to tuple
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

    if parse_key:
        new_result = {}
        for key, value in result.items():
            new_key = tuple(key.split('/'))
            new_result[new_key] = value
        result = new_result

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

def df_to_xls(df, save_path):
    """Save single pd.DataFrame to a excel file"""
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    writer = pd.ExcelWriter(save_path)
    df.to_excel(writer)
    writer.save()

def df_to_pickle(df, save_path):
    """Save single pd.DataFrame to a pickle file"""
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_pickle(save_path)

def dfs_to_xls(dfs, save_path):
    """Save multiple pd.DataFrame in a dict to a excel file
    
    Args:
        dfs (dict): {sheet_name: pd.DataFrame}
    """
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    writer = pd.ExcelWriter(save_path)
    for k, df in dfs.items():
        df.to_excel(writer, '%s'%k)
    writer.save()

def dict_to_xls(dic, row_keys_idx, col_keys_idx, save_path, sheet_idx=None):
    """Convert dict to excel
    
    Args:
        dic: result dictionary. Keys are tuples.
        row_keys_idx: index of keys for rows, ordered hierarchicallly
        col_keys_idx: index of keys for columns, ordered hierarchicallly
        sheet_idx: index of keys for sheet
    """
    if sheet_idx is None:
        df = dict_to_df(dic, row_keys_idx, col_keys_idx)
        df_to_xls(df, save_path)
    else:
        dfs = dict_to_dfs(dic, row_keys_idx, col_keys_idx, sheet_idx)
        dfs_to_xls(dfs, save_path)

def flatten_dict(dictionary):
    """Convert hierarchic dictionary into a flat dict by extending dimension of keys.
    (e.g. {"a": {"b":"c"}} -> {("a", "b"): "c"})
    """
    values = list(dictionary.values())
    if any([type(v) != dict for v in values]):
        return dictionary

    flat_dict = {}
    for k, v in dictionary.items():
        if type(k) != tuple:
            k = (k,)
        for vk, vv in v.items():
            if type(vk) != tuple:
                vk = (vk,)
            new_key = k + vk
            flat_dict[new_key] = vv
    return flatten_dict(flat_dict)

def rearrange_dict(dictionary, order):
    """Rearrange the order of key of dictionary"""
    new_dict = {}
    for key, value in dictionary.items():
        if len(key) < len(order):
            print("Number of new order must be smaller than the length of key")
            sys.exit()

        new_order = np.arange(len(key))
        for i, o in enumerate(order):
            new_order[i] = o

        new_key = tuple([key[i] for i in new_order])
        new_dict[new_key] = value
    return new_dict

def makedirs(dir_list):
    save_dir = os.path.join(*dir_list)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def result_to_table(result, save_dir, csv=True, xls=True):
    """Convert result to tables. One table for each dataset.

    Args:
        result (dict): key: (dataset_name, split_seed, error_type, train_file, model_name, seed)
        csv (bool): save csv table
        xls (bool): save xls table

    """

    # save csv table
    if csv:
        csv_dir = makedirs([save_dir, 'csv'])
        flat_result = flatten_dict({k + ('result',):v for k, v in result.items()})
        result_df = dict_to_df(flat_result, [0, 1, 2, 3, 4, 5, 7], [6])
        save_path = os.path.join(csv_dir, "training_result.csv")
        result_df.to_csv(save_path, index_label=['dataset', 'split_seed', 'error_type', 'train_file', 'model_name', 'seed', 'metric'])

    if xls:
        xls_dir = makedirs([save_dir, 'xls'])
        datasets = list({k[0] for k in result.keys()})

        for dataset in datasets:
            dataset_result = flatten_dict({k:v for k, v in result.items() if k[0] == dataset})
            save_path = os.path.join(xls_dir, '{}_result.xls'.format(dataset))
            dict_to_xls(dataset_result, [0, 1, 3, 4, 5], [6], save_path, sheet_idx=2)
            
def group(result, idx, keepdim=False):
    """Group results on one dimension (key component) into a list

    Args:
        result (dict): result dict
            key (tuple): e.g. (dataset_name, split_seed, error_type, train_file, model_name, seed)
            value (dict): {metric_name: metric}
        idx: the index of dimension (key component) by which the result is grouped 
        keepdim (bool): keep or delete dimension by which the result is grouped 
    """

    # get domain in given dimension (key component)
    domain = list({k[idx] for k in result.keys()})

    # loop through each value in domain, append corresponding results into a list
    new_result = {}
    for x in domain:
        for old_key, v in result.items():

            if x != old_key[idx]:
                continue

            # new key (eliminate the given dimension)
            new_key = tuple([old_key[i] for i in range(len(old_key)) if i != idx])

            # new value 
            if new_key not in new_result.keys():
                new_result[new_key] = defaultdict(list)
            
            # apppend results into list
            for vk, vv in v.items():
                # don't include best param saved in result
                if vk != "best_params":
                    new_result[new_key][vk].append(vv)

            if keepdim:
                new_result[new_key]["group_key"].append(old_key[idx])
    
    if keepdim:
        final_result = {}
        for k, v in new_result.items():
            group_key = "/".join(v["group_key"])
            new_k = k[0:idx] + (group_key,) + k[idx:]
            del v["group_key"]
            final_result[new_k] = v
        new_result = final_result
    return new_result

def reduce_by_mean(result):
    """Reduce a list of results into a single result by mean
        
    Args:
        result (dict): result dict
            key (tuple): (dataset_name, split_seed, error_type, train_file, model_name)
            value (dict): {metric_name: [metric lists]}
    """
    new_result = {}
    for k, v in result.items():
        new_value = {}
        for vk, vv in v.items():
            new_value[vk] = np.mean(vv)
        new_result[k] = new_value
    return new_result

def reduce_by_max_val(result, dim=None, dim_name=None):
    """Reduce a list of results into a single result by the result corresponding to the best val_acc
        
    Args:
        result (dict): result dict
            key (tuple): (dataset_name, split_seed, error_type, train_file, model_name)
            value (dict): {metric_name: [metric lists]}
    """
    new_result = {}
    for k, v in result.items():
        new_value = {}

        if np.isnan(v['val_acc']).all():
            best_val_idx = 0
        else:
            best_val_idx = np.nanargmax(v['val_acc'])

        if dim is not None:
            best = k[dim].split('/')[best_val_idx]
            new_key = k[0:dim] + (dim_name,) + k[dim+1:]
        else:
            new_key = k

        for vk, vv in v.items():
            new_value[vk] = vv[best_val_idx]

        if dim is not None:
            new_value[dim_name] = best

        new_result[new_key] = new_value

    return new_result

def group_reduce_by_best_clean(result):
    """Group by clean method and then reduce a list of results into a single result by the result corresponding to the best val_acc
        
    Args:
        result (dict): result dict
            key (tuple): (dataset_name, split_seed, error_type, train_file, model_name)
            value (dict): {metric_name: [metric lists]}
    """
    dirty = {}
    clean = {}
    for k, v in result.items():
        train_file = k[3]
        if train_file[0:5] == "dirty" or train_file[0:5] == "delet":
            dirty[k] = v
        else:
            new_v = {}
            for vk, vv in v.items():
                vk_list = vk.split('_')
                if vk_list[0] in ['clean', 'impute']:
                    new_vk = '_'.join([vk_list[0], vk_list[-2], vk_list[-1]])
                else:
                    new_vk = vk                
                
                new_v[new_vk] = vv
            clean[k] = new_v

    clean = group(clean, 3, keepdim=True)
    clean = reduce_by_max_val(clean, dim=3, dim_name="clean")

    new_clean = {}
    for k, v in clean.items():
        if k[2] == 'missing_values':
            new_k = k[0:3] + ('impute',) + k[4:]
        else:
            new_k = k
        new_clean[new_k] = v
    
    new_dirty = {}
    for k, v in dirty.items():
        new_v = {}
        clean_key = k[0:3] + ("clean",) + k[4:]
        clean_method = clean[clean_key]["clean"]

        new_v = {}
        for vk, vv in v.items():
            vk_list = vk.split('_')
            if vk_list[0] not in ['clean', 'impute']:
                new_v[vk] = vv

        if k[2] == 'missing_values':
            new_v["impute_test_acc"] = v["{}_test_acc".format(clean_method)]
            new_v["impute_test_f1"] = v["{}_test_f1".format(clean_method)]
        else:
            new_v["clean_test_acc"] = v["{}_test_acc".format(clean_method)]
            new_v["clean_test_f1"] = v["{}_test_f1".format(clean_method)]

        new_dirty[k] = new_v

    new_result = {**new_dirty, **new_clean}
    return new_result