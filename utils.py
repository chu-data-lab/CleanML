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
            df[cat] = df[cat].astype(str)
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
        return [train_file, "dirty"]

    if error_type == "missing_values":
        return ["dirty", 
                "clean_impute_mean_mode", 
                "clean_impute_mean_dummy", 
                "clean_impute_median_mode", 
                "clean_impute_median_dummy", 
                "clean_impute_mode_mode", 
                "clean_impute_mode_dummy"]
    elif error_type == "outliers":
        return ["dirty", 
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
        return ["dirty", "clean"]
