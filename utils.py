import pandas as pd
import os
import config
import sys
import json
import model
import numpy as np

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

def load_df(dataset, file_dir):
    """
    Get pandas data frame and preprocess categorical variables

    Args: 
        dataset: dataset in config.py
        file_dir: directory of data file
    Return:
        df: pandas data frame
    """
    df = pd.read_csv(file_dir)
    if 'categorical_variables' in dataset.keys():
        categories = dataset['categorical_variables']
        for cat in categories:
            df[cat] = df[cat].astype(str).replace('nan', np.nan)
    return df

def load_dfs(dataset, file_dir_pfx):
    train_dir = file_dir_pfx + '_train.csv'
    test_dir = file_dir_pfx + '_test.csv'
    train = load_df(dataset, train_dir)
    test = load_df(dataset, test_dir)
    return train, test

def save_dfs(train, test, save_dir_pfx):
    train_save_dir = save_dir_pfx + '_train.csv'
    test_save_dir = save_dir_pfx + '_test.csv'
    train.to_csv(train_save_dir, index=False)
    test.to_csv(test_save_dir, index=False)

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
    data_dir = os.path.join(config.root_dir, dataset['data_dir'])
    if folder is None:
        return data_dir

    folder_dir = os.path.join(data_dir, folder)
    if create_folder and not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

    if file is None:
        return folder_dir
    
    file_dir = os.path.join(folder_dir, file)
    return file_dir

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
    models = model.models
    mod = [m for m in models if m['name'] == model_name ]
    if len(mod) == 0:
        print("Invalid model name: {}".format(model_name))
        sys.exit()
    return mod[0]

def get_test_files(error_type, train_file):
    file_type = train_file[0:5]
    if file_type == "clean":
        return ["dirty", train_file]
    else:
        return get_filenames(error_type)

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

def get_filenames(error_type):
    if error_type == 'missing_values':
        filenames = ["dirty", 
                    "clean_impute_mean_mode", 
                    "clean_impute_mean_dummy", 
                    "clean_impute_median_mode", 
                    "clean_impute_median_dummy", 
                    "clean_impute_mode_mode", 
                    "clean_impute_mode_dummy"]
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
                     "clean_iso_forest_impute_median_dummy"]
    else:
        filenames = ["dirty", "clean"]
    return filenames

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

def dict_to_xls(dic, row_keys_idx, col_keys_idx, sheet_idx, save_dir):
    """ Convert dict to excel
    
    Args:
        dic: result dictionary. Keys are tuples.
        row_keys_idx: index of keys for rows, ordered hierarchicallly
        col_keys_idx: index of keys for columns, ordered hierarchicallly
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

def combine(result):
    # Combine result from different experiments into a list
    seeds = list({k.split('/')[4] for k in result.keys()})
    comb = {}
    for s in seeds:
        for k, v in result.items():
            dataset, error, file, model, seed = k.split('/')
            if s != seed:
                continue

            key = (dataset, error, file, model)
            value = {vk:[vv] for vk, vv in v.items()}

            if key not in comb.keys():
                comb[key] = value
            else:
                for vk, vv in v.items():
                    comb[key][vk].append(vv)
    return comb

def replace_result(result1_dir, result2_dir, dataset_name):
    result1 = json.load(open(result1_dir, 'r'))
    result2 = json.load(open(result2_dir, 'r'))
    rep = {k:v for k, v in result2.items() if dataset_name == k.split('/')[0]}
    result = {**result1, **rep}
    json.dump(result, open('./result.json', 'w'))