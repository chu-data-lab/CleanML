import pandas as pd
import os
import config
import sys

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

def get_df(dataset, file_dir):
    """
    Get pandas data frame and preprocess categorical variables

    Args: 
        dataset: dataset in config.py
        file_dir: directory of data file
    Return:
        df: pandas data frame
    """
    df = pd.read_csv(file_dir)
    categories = dataset['categorical_variables']
    for cat in categories:
        df[cat] = df[cat].astype(str)
    return df

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