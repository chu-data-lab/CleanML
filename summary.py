import numpy as np
import pandas as pd
import config
import os
import sys
import re
import utils

def get_indicator_dirs(dataset, error_type):
    save_dir = utils.get_dir(dataset, error_type)
    regex = re.compile(r'indicator.*train')
    filenames = filter(regex.search, os.listdir(save_dir))
    indicator_dirs = [utils.get_dir(dataset, error_type, file) for file in filenames]
    return indicator_dirs

def count_error(indicator_dir, error_type):
    filename = os.path.split(indicator_dir)[-1][10:-4]
    indicator = pd.read_csv(indicator_dir)
    N, m = indicator.shape
    n_error_record = indicator.any(axis = 1).sum()
    error_rate_record = n_error_record / N

    if m > 1:
        n_entry = N * m
        n_error_entry = np.sum(indicator.values)
        error_rate_entry = n_error_entry / n_entry
        count_summary = 'Number of {} in {}: {}/{} ({:.2%}) entries, {}/{} ({:.2%}) records'\
                    .format(error_type, filename, n_error_entry, n_entry, error_rate_entry, n_error_record, N, error_rate_record)
    else:
        count_summary = 'Number of {} in {}: {}/{} ({:.2%}) records'\
                    .format(error_type, filename, n_error_record, N, error_rate_record)
    return count_summary


datasets = config.datasets
summary = "The summary of error in each dataset:\n\n"

for dataset in datasets:
    summary += '-------------------------------------------------\n'
    summary += '[{}]\n'.format(dataset['data_dir'])

    for error in dataset['error_types']:
        summary += '- {}:\n'.format(error)
        indicator_dirs = get_indicator_dirs(dataset, error)
        for indicator_dir in indicator_dirs:
            summary += '  {}\n'.format(count_error(indicator_dir, error))
        
print(summary)


