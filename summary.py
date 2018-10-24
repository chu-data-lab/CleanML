import numpy as np
import pandas as pd
import config
import os
import sys
import re
import utils

def get_dir(dataset, error_type, source='raw'):
    data_dir = os.path.join(root_dir, dataset['data_dir'])
    save_dir = os.path.join(data_dir, error_type)
    return save_dir

def count_error(indicator, error_type):
    N, m = indicator.shape
    is_error = indicator.any(axis = 1)
    num_error = is_error.sum()
    count_summary = 'Number of {}: {}/{} ({:.2%})'.format(error_type, num_error, N, num_error/N)
    return count_summary

root_dir = config.root_dir
datasets = config.datasets

summary = "The summary of error in each dataset:\n\n"

for dataset in datasets:
    summary += '-------------------------------------------------\n'
    summary += '[{}]\n'.format(dataset['data_dir'])

    for error in dataset['error_types']:
        if error == 'mv':
            summary += '- Missing Values:\n'
            save_dir = get_dir(dataset, 'missing_values')
            indicator_dir = os.path.join(save_dir, 'indicator.csv')
            indicator = pd.read_csv(indicator_dir)
            summary += '  {}\n'.format(count_error(indicator, 'Missing Values'))

        if error == 'out':
            summary += '- Outliers:\n'
            save_dir = utils.get_dir(dataset, 'outliers')
            regex = re.compile(r'indicator')
            filenames = filter(regex.search, os.listdir(save_dir))

            for filename in filenames:
                name, _ = filename.split('.')
                _, method = name.split('_', 1)
                
                indicator_dir = os.path.join(save_dir, filename)
                summary += '  {}\n'.format(count_error(indicator, 'Outliers ({})'.format(method)))

        if error == 'dup':
            summary += '- Duplicates:\n'
            save_dir = get_dir(dataset, 'duplicates')
            indicator_dir = os.path.join(save_dir, 'indicator.csv')
            indicator = pd.read_csv(indicator_dir)
            summary += '  {}\n'.format(count_error(indicator, 'Duplicates'))

        if error == 'incon':
            summary += '- Inconsistency:\n'
            save_dir = get_dir(dataset, 'inconsistency')
            dirty_dir = os.path.join(save_dir, 'dirty.csv')
            clean_dir = os.path.join(save_dir, 'clean.csv')
            dirty = pd.read_csv(dirty_dir)
            clean = pd.read_csv(clean_dir)
            indicator = (dirty.values != clean.values)
            summary += '  {}\n'.format(count_error(indicator, 'Inconsistency'))
        
print(summary)


