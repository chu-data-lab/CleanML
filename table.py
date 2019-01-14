""" Generate tables from results"""
import json
import pandas as pd
import numpy as np
import utils
from collections import defaultdict

def group_by_seed(result):
    """ Combine result from different experiments into a list

        Args:
            result (dict): result dict
                             key (string): dataset_name/error_type/train_file/model_name/seed
                             value (dict): metric_name: metric
    """
    # get all seeds
    seeds = list({k.split('/')[4] for k in result.keys()})
    
    new_result = {}
    for s in seeds:
        for k, v in result.items():
            dataset, error, file, model, seed = k.split('/')
            if s == seed:
                # new key
                key = (dataset, error, file, model)

                # new value
                if key not in new_result.keys():
                    new_result[key] = defaultdict(list)
                
                # apppend results
                for vk, vv in v.items():
                    # don't include best param and seeds saved in result
                    if vk not in ["best_params", "seeds"]:
                        new_result[key][vk].append(vv)
    return new_result

def reduce_by_mean(result):
    """ Reduce a list of results from different experiments into a single result by mean
        
        Args:
        result (dict): result dict
                         key (tuple): (dataset_name, error_type, train_file, model_name)
                         value (dict): metric_name: metric lists
    """
    new_result = {}
    for k, v in result.items():
        new_value = {}
        for vk, vv in v.items():
            new_value[vk] = np.mean(vv)
        new_result[k] = new_value
    return new_result

def is_metric_f1(dataset_name):
    dataset = utils.get_dataset(dataset_name)
    return ('class_imbalance' in dataset.keys() and dataset['class_imbalance'])

def get_metric_name(dataset_name, test_file):
    if is_metric_f1(dataset_name):
        metric = test_file + "_test_f1"
    else:
        metric = test_file + "_test_acc"
    return metric

def get_four_metrics(result, error_type, file_types):
    """ Get four metrics for all datasets in a table (pandas.DataFrame)

        Args:
            result (dict): result dict
            error_type (string): error type
            file_types (list): names of two types of train or test files
    """

    four_metrics = {}
    for (dataset, error, train_file, model), v in result.items():
        if error == error_type and train_file in file_types:
            for test_file in file_types:
                metric_name = get_metric_name(dataset, test_file)
                metric = v[metric_name]
                four_metrics[(dataset, train_file, model, test_file)] = metric
    four_metrics = utils.dict_to_df(four_metrics, [0, 1], [2, 3])
    return four_metrics

def compare_four_metrics(four_metrics, file_types):
    """ Compute the relative difference between four metrics

        Args:
            four_metrics (pandas.DataFrame): four metrics
            file_types (list): names of two types of train or test files

    """
    A = lambda m: m.loc[file_types[0], file_types[0]]
    B = lambda m: m.loc[file_types[0], file_types[1]]
    C = lambda m: m.loc[file_types[1], file_types[0]]
    D = lambda m: m.loc[file_types[1], file_types[1]]
    dAB = lambda m: (B(m) - A(m)) / A(m)
    dCD = lambda m: (D(m) - C(m)) / C(m)
    dAC = lambda m: (C(m) - A(m)) / A(m)
    dBD = lambda m: (D(m) - B(m)) / B(m)

    comparison = {}
    datasets = list(set(four_metrics.index.get_level_values(0)))
    models = list(set(four_metrics.columns.get_level_values(0)))
    for dataset in datasets:
        for model in models:
            m = four_metrics.loc[dataset, model]
            comparison[(dataset, model, "AB")] = dAB(m)
            comparison[(dataset, model, "CD")] = dCD(m)
            comparison[(dataset, model, "AC")] = dAC(m)
            comparison[(dataset, model, "BD")] = dBD(m)
    comparison = utils.dict_to_df(comparison, [0, 1], [2])
    return comparison

def compare_dup_incon(result, error):
    """ Comparison for duplicates and inconsistency
        Args:
            error: "duplicates or inconsistency"

    """
    file_types = ["dirty", "clean"]
    four_metrics = get_four_metrics(result, error, file_types)
    comparison = compare_four_metrics(four_metrics, file_types)
    comparisons = {"clean": comparison}
    return comparisons

def compare_out(result):
    """ Comparison for outliers"""
    clean_methods = ["clean_SD_delete", "clean_SD_impute_mean_dummy", "clean_SD_impute_median_dummy",
                    "clean_IQR_delete", "clean_IQR_impute_mean_dummy", "clean_IQR_impute_median_dummy",
                    "clean_iso_forest_delete", "clean_iso_forest_impute_mean_dummy", "clean_iso_forest_impute_median_dummy"]
    comparisons = {}
    for method in clean_methods:
        file_types = ['dirty', method]
        four_metrics = get_four_metrics(result, "outliers", file_types)
        comparison = compare_four_metrics(four_metrics, file_types)
        comparisons[method] = comparison
    return comparisons

def compare_mv(result):
    """ Comparison for missing values"""
    impute_methods = ["clean_impute_mean_mode", "clean_impute_mean_dummy", 
                      "clean_impute_median_mode", "clean_impute_median_dummy", 
                      "clean_impute_mode_mode", "clean_impute_mode_dummy"]
    comparisons = {}
    for method in impute_methods:
        file_types = ['dirty', method]
        four_metrics = get_four_metrics(result, "missing_values", file_types)
        comparison = compare_four_metrics(four_metrics, file_types)
        comparisons[method] = comparison
    return comparisons

def compare_mislabel(result):
    """ Comparison for mislabel"""
    inject_methods = ["dirty_uniform", "dirty_major", "dirty_minor"]

    comparisons = {}
    for method in inject_methods:
        file_types = [method, "clean"]
        four_metrics = get_four_metrics(result, "mislabel", file_types)
        comparison = compare_four_metrics(four_metrics, file_types)
        comparisons[method] = comparison
    return comparisons