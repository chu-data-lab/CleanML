""" Generate tables from results"""
import json
import pandas as pd
import numpy as np
import utils
from collections import defaultdict
from scipy.stats import ttest_rel
import config
import os
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests

def group(result, idx):
    """ Combine result from different experiments into a list

        Args:
            result (dict): result dict
                             key (string): dataset_name/split_seed/error_type/train_file/model_name/seed
                             value (dict): metric_name: metric
            idx: the index of key component by which the result is grouped 
    """
    # get all seeds
    seeds = list({k.split('/')[idx] for k in result.keys()})
    
    new_result = {}
    for s in seeds:
        for k, v in result.items():
            old_key = k.split('/')
            seed = old_key[idx]

            if s == seed:
                # new key
                key = tuple([old_key[i] for i in range(len(old_key)) if i != idx])

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
    for (dataset, split_seed, error, train_file, model), v in result.items():
        if error == error_type and train_file in file_types:
            for test_file in file_types:
                metric_name = get_metric_name(dataset, test_file)
                metric = v[metric_name]
                four_metrics[(dataset, split_seed, train_file, model, test_file)] = metric
    four_metrics = utils.dict_to_df(four_metrics, [0, 2, 1], [3, 4])
    return four_metrics

def compare_four_metrics(four_metrics, file_types, compare_method):
    """ Compute the relative difference between four metrics

        Args:
            four_metrics (pandas.DataFrame): four metrics
            file_types (list): names of two types of train or test files
            compare_method (fn): function to compare two metrics
    """
    A = lambda m: m.loc[file_types[0], file_types[0]]
    B = lambda m: m.loc[file_types[0], file_types[1]]
    C = lambda m: m.loc[file_types[1], file_types[0]]
    D = lambda m: m.loc[file_types[1], file_types[1]]
    AB = lambda m: compare_method(A(m), B(m))
    AC = lambda m: compare_method(A(m), C(m))
    CD = lambda m: compare_method(C(m), D(m))
    BD = lambda m: compare_method(B(m), D(m))

    comparison = {}
    datasets = list(set(four_metrics.index.get_level_values(0)))
    models = list(set(four_metrics.columns.get_level_values(0)))
    for dataset in datasets:
        for model in models:
            m = four_metrics.loc[dataset, model]
            comparison[(dataset, model, "AB")] = AB(m)
            comparison[(dataset, model, "CD")] = CD(m)
            comparison[(dataset, model, "AC")] = AC(m)
            comparison[(dataset, model, "BD")] = BD(m)

    # comparison = utils.dict_to_df(comparison, [0, 1], [2])
    return comparison

def two_tailed_t_test(dirty, clean):
    n_d = len(dirty)
    n_c = len(clean)
    n = min(n_d, n_c)
    t, p = ttest_rel(clean[:n], dirty[:n])
    if np.isnan(t):
        t, p = 0, 1
    return {"t-stats":t, "p-value":p}

def one_tailed_t_test(dirty, clean, direction):
    two_tail = two_tailed_t_test(dirty, clean)
    t, p_two = two_tail['t-stats'], two_tail['p-value']
    if direction == 'positive':
        if t > 0 :
            p = p_two * 0.5
        else:
            p = 1 - p_two * 0.5
    else:
        if t < 0:
            p = p_two * 0.5
        else:
            p = 1 - p_two * 0.5
    return {"t-stats":t, "p-value":p}

def t_test(dirty, clean):
    result = {}
    result['two_tailed_t_test'] = two_tailed_t_test(dirty, clean)
    result['one_tailed_t_test_pos'] = one_tailed_t_test(dirty, clean, 'positive')
    result['one_tailed_t_test_neg'] = one_tailed_t_test(dirty, clean, 'negative')
    return result

def compare_dup_incon(result, error, compare_method):
    """ Comparison for duplicates and inconsistency
        Args:
            error: "duplicates or inconsistency"

    """
    file_types = ["dirty", "clean"]
    four_metrics = get_four_metrics(result, error, file_types)
    comparison = compare_four_metrics(four_metrics, file_types, compare_method)
    comparisons = {"clean": comparison}
    metrics = {"clean": four_metrics}
    return comparisons, metrics

def compare_out(result, compare_method):
    """ Comparison for outliers"""
    clean_methods = ["clean_SD_delete", "clean_SD_impute_mean_dummy", "clean_SD_impute_median_dummy", "clean_SD_impute_mode_dummy",
                    "clean_IQR_delete", "clean_IQR_impute_mean_dummy", "clean_IQR_impute_median_dummy", "clean_IQR_impute_mode_dummy", 
                    "clean_iso_forest_delete", "clean_iso_forest_impute_mean_dummy", "clean_iso_forest_impute_median_dummy", "clean_iso_forest_impute_mode_dummy"]
    metrics = {}
    comparisons = {}
    for method in clean_methods:
        file_types = ['dirty', method]
        four_metrics = get_four_metrics(result, "outliers", file_types)
        comparison = compare_four_metrics(four_metrics, file_types, compare_method)
        metrics[method] = four_metrics
        comparisons[method] = comparison
    return comparisons, metrics

def compare_mv(result, compare_method):
    """ Comparison for missing values"""
    impute_methods = ["impute_mean_mode", "impute_mean_dummy", 
                      "impute_median_mode", "impute_median_dummy", 
                      "impute_mode_mode", "impute_mode_dummy"]
    metrics = {}
    comparisons = {}
    for method in impute_methods:
        file_types = ['delete', method]
        four_metrics = get_four_metrics(result, "missing_values", file_types)
        comparison = compare_four_metrics(four_metrics, file_types, compare_method)
        metrics[method] = four_metrics
        comparisons[method] = comparison
    return comparisons, metrics

def compare_mislabel(result, compare_method):
    """ Comparison for mislabel"""
    inject_methods = ["dirty_uniform", "dirty_major", "dirty_minor"]

    comparisons = {}
    metrics = {}
    for method in inject_methods:
        file_types = [method, "clean"]
        four_metrics = get_four_metrics(result, "mislabel", file_types)
        comparison = compare_four_metrics(four_metrics, file_types, compare_method)
        metrics[method] = four_metrics
        comparisons[method] = comparison
    return comparisons, metrics

def save_dfs(dfs, save_dir):
    directory = os.path.dirname(save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    writer = pd.ExcelWriter(save_dir)
    for k, df in dfs.items():
        if "iso_forest" in k:
            k = k.replace("iso_forest", "ISO")
        df.to_excel(writer, '%s'%k)
    writer.save()

def save_t_test(t_test_results, save_dir, two_tailed_reject, one_tailed_pos_reject, one_tailed_neg_reject):
    directory = os.path.dirname(save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    writer = pd.ExcelWriter(save_dir, engine='xlsxwriter')
    t_test_results.to_excel(writer, sheet_name='t-test')
    workbook = writer.book
    worksheet = writer.sheets['t-test']
    cell_format = workbook.add_format({'bold': True, 'font_color': 'red'})

    for i, r in enumerate(two_tailed_reject):
        if r:
            p = t_test_results.loc[:, ("two_tailed_t_test", 'p-value')][i]
            worksheet.write(3 + i, 6, p, cell_format)
    for i, r in enumerate(one_tailed_pos_reject):
        if r:
            p = t_test_results.loc[:, ("one_tailed_t_test_pos", 'p-value')][i]
            worksheet.write(3 + i, 8, p, cell_format)
    for i, r in enumerate(one_tailed_neg_reject):
        if r:
            p = t_test_results.loc[:, ("one_tailed_t_test_neg", 'p-value')][i]
            worksheet.write(3 + i, 10, p, cell_format)

    writer.save()


def flatten_dict(dictionary):
    flat_dict = {}
    if type(list(dictionary.values())[0]) != dict:
        return dictionary
            
    for k, v in dictionary.items():
        if type(k) != tuple:
            k = (k,)
        for vk, vv in v.items():
            if type(vk) != tuple:
                vk = (vk,)
            new_key = k + vk
            flat_dict[new_key] = vv
    return flatten_dict(flat_dict)

def BY_procedure(t_test_results, test_type, alpha=0.05):
    p_vals = t_test_results.loc[:, (test_type, 'p-value')]
    reject, _, _, _ = multipletests(p_vals, method='fdr_by', alpha=alpha)
    reject_df = pd.DataFrame(reject, index=p_vals.index, columns=['reject'])
    a = p_vals.values[reject].max()
    return reject_df, a

# def count_metrics(reject):
#     cases = ["AB", "CD", "AC", "BD"]
#     idx = pd.IndexSlice
#     count = {}

#     for c in cases:
#         reject_c = reject.loc[idx[:,:,:,:,c], :].values
#         count[c] = "{}/{}".format(np.sum(reject_c), len(reject_c))
#     return count

def count_reject_one_dim(reject, index):
    cases = reject.index.get_level_values(index)
    idx = pd.IndexSlice
    if index == 0:
        idxs = [idx[c,:,:,:,:] for c in cases]
    if index == 1:
        idxs = [idx[:,c,:,:,:] for c in cases]
    if index == 2:
        idxs = [idx[:,:,c,:,:] for c in cases]
    if index == 3:
        idxs = [idx[:,:,:,c,:] for c in cases]
    if index == 4:
        idxs = [idx[:,:,:,:,c] for c in cases]
    count = {}

    for i, c in zip(idxs, cases):
        reject_c = reject.loc[i, :].values
        count[c] = "{}/{} ({:.0%})".format(np.sum(reject_c), len(reject_c), np.sum(reject_c)/len(reject_c))
    return count

def count_reject_general(rejects, writer):
    dimension = ['error_types', 'datasets', 'clean_methods', 'ml models', 'scenarios']
    row = 0
    for i, d in enumerate(dimension):
        if i == 2:
            continue
        count = {}
        for name, reject in rejects.items():
            count[name] = count_reject_one_dim(reject, i)
        result = flatten_dict({d: count})
        result = utils.dict_to_df(result, [1], [0, 2])
        result.to_excel(writer, sheet_name='general', startrow=row, startcol=0) 
        row += 10

def count_reject_error(rejects, writer, error):
    dimension = ['error_types', 'datasets', 'clean_methods', 'ml models', 'scenarios']
    row = 0
    for i, d in enumerate(dimension):
        if i == 0:
            continue
        count = {}
        for name, reject in rejects.items():
            idx = pd.IndexSlice
            count[name] = count_reject_one_dim(reject.loc[idx[error, :, :, :, :], :], i)
        result = flatten_dict({d: count})
        result = utils.dict_to_df(result, [1], [0, 2])
        result.to_excel(writer, sheet_name=error, startrow=row, startcol=0) 
        row += 10

def conduct_t_test(result, save_metrics=False):
    t_test_results = {}
    mv_comp, mv_metrics = compare_mv(result, t_test)
    out_comp, out_metrics = compare_out(result, t_test)
    ml_comp,  ml_metrics = compare_mislabel(result, t_test)
    dup_comp, dup_metrics = compare_dup_incon(result, "duplicates", t_test)
    incon_comp, incon_metrics = compare_dup_incon(result, "inconsistency", t_test)

    t_test_results['missing_values'] = mv_comp
    t_test_results['outliers'] = out_comp
    t_test_results['mislabel'] = ml_comp
    t_test_results['duplicates'] = dup_comp
    t_test_results['inconsistency'] = incon_comp

    t_test_results = flatten_dict(t_test_results)
    t_test_results = utils.dict_to_df(t_test_results, [0, 2, 1, 3, 4], [5, 6])

    if save_metrics:
        save_dfs(mv_metrics, "./table/four_metrics/missing_values_four_metrics.xls")
        save_dfs(out_metrics, "./table/four_metrics/outliers_four_metrics.xls")
        save_dfs(ml_metrics, "./table/four_metrics/mislabel_four_metrics.xls")
        save_dfs(dup_metrics, "./table/four_metrics/duplicates_four_metrics.xls")
        save_dfs(incon_metrics, "./table/four_metrics/inconsistency_four_metrics.xls")
    return t_test_results


if __name__ == '__main__':
    # result = utils.load_result()
    # result = group(result, 5)
    # result = reduce_by_mean(result)
    # t_test_results = conduct_t_test(result)

    # two_tailed_reject, two_tailed_a = BY_procedure(t_test_results, 'two_tailed_t_test')
    # one_tailed_pos_reject, one_tailed_pos_a = BY_procedure(t_test_results, 'one_tailed_t_test_pos')
    # one_tailed_neg_reject, one_tailed_neg_a  = BY_procedure(t_test_results, 'one_tailed_t_test_neg')
    # print(two_tailed_a, one_tailed_pos_a, one_tailed_neg_a)
    # two_tailed_reject.to_pickle('./table/t_test/two_tailed_reject.pkl')
    # one_tailed_pos_reject.to_pickle('./table/t_test/one_tailed_pos_reject.pkl')
    # one_tailed_neg_reject.to_pickle('./table/t_test/one_tailed_neg_reject.pkl')

    two_tailed_reject = pd.read_pickle('./table/t_test/two_tailed_reject.pkl')
    one_tailed_pos_reject = pd.read_pickle('./table/t_test/one_tailed_pos_reject.pkl')
    one_tailed_neg_reject = pd.read_pickle('./table/t_test/one_tailed_neg_reject.pkl')

    rejects = {"two_tailed": two_tailed_reject, "one_tailed_pos": one_tailed_pos_reject, "one_tailed_neg": one_tailed_neg_reject}
    writer = pd.ExcelWriter('./table/t_test/count.xlsx',engine='xlsxwriter')   
    workbook = writer.book
    
    worksheet = workbook.add_worksheet('general')
    writer.sheets['general'] = worksheet
    count_reject_general(rejects, writer)

    for error in ['missing_values', 'outliers', 'mislabel']:
        worksheet = workbook.add_worksheet(error)
        writer.sheets[error] = worksheet
        idx = pd.IndexSlice
        count_reject_error(rejects, writer, error)

    writer.save()
   

    # print(AB.shape)
    # print(np.sum(two_tailed_reject), np.sum(one_tailed_pos_reject), np.sum(one_tailed_neg_reject))
    


    # save_t_test(t_test_results, "./table/t_test/t_test_results.xlsx", two_tailed_reject, one_tailed_pos_reject, one_tailed_neg_reject)
    




