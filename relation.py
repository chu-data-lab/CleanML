"""Populate relations using training results"""
import json
import pandas as pd
import numpy as np
import utils
from scipy.stats import ttest_rel
import config
import os
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests, fdrcorrection_twostage
import json
import sys

"""Compare class"""
class Compare(object):
    def __init__(self, result, compare_method, compare_metric):
        super(Compare, self).__init__()
        """ Compare
        Args:
            result (dict): result dict
            compare_method (fn): function to compare two metrics
            compare_metric (fn): function to specify metrics to be compared
        """
        self.result = result
        self.compare_metric = compare_metric
        self.compare_method = compare_method

        self.four_metrics = {}
        self.compare_result = {}
        for error_type in config.error_types:
            self.compare_result[error_type['name']], self.four_metrics[error_type['name']] = self.compare_error(error_type['name'])
        
        # key order: error/clean_method/dataset/models/scenario/ [compare_keys...]
        self.compare_result = utils.flatten_dict(self.compare_result)

        # rearrange key order: error/dataset/clean_method/models/scenario/ [compare_keys...]
        self.compare_result = utils.rearrange_dict(self.compare_result, [0, 2, 1, 3, 4])

    def get_four_metrics(self, error_type, file_types):
        """Get four metrics (A, B, C, D) for all datasets in a table (pd.DataFrame)

        Args:
            error_type (string): error type
            file_types (list): names of two types of train or test files
        """
        four_metrics = {}
        for (dataset, split_seed, error, train_file, model), value in self.result.items():
            if error == error_type and train_file in file_types:
                for test_file in file_types:
                    metric_name = self.compare_metric(dataset, error_type, test_file)
                    metric = value[metric_name]
                    four_metrics[(dataset, split_seed, train_file, model, test_file)] = metric

        four_metrics = utils.dict_to_df(four_metrics, [0, 2, 1], [3, 4]).sort_index()
        return four_metrics

    def compare_four_metrics(self, error_type, four_metrics, file_types):
        """Compute the relative difference between four metrics

        Args:
            four_metrics (pandas.DataFrame): four metrics
            file_types (list): names of two types of train or test files
            compare_method (fn): function to compare two metrics
        """
        A = lambda m: m.loc[file_types[0], file_types[0]]
        B = lambda m: m.loc[file_types[0], file_types[1]]
        C = lambda m: m.loc[file_types[1], file_types[0]]
        D = lambda m: m.loc[file_types[1], file_types[1]]

        scenarios = {
            "CD":lambda m: self.compare_method(C(m), D(m)),
            "BD":lambda m: self.compare_method(B(m), D(m)),
            "AB":lambda m: self.compare_method(A(m), B(m)),
            "AC":lambda m: self.compare_method(A(m), C(m))
        }

        comparison = {}
        datasets = list(set(four_metrics.index.get_level_values(0)))
        models = list(set(four_metrics.columns.get_level_values(0)))
        for dataset in datasets:
            for model in models:
                m = four_metrics.loc[dataset, model]
                for s in config.scenarios[error_type]:
                    comparison[(dataset, model, s)] = scenarios[s](m)
        # comparison = utils.dict_to_df(comparison, [0, 1], [2])
        return comparison

    def compare_error(self, error_type):
        """Compare four metrics based on compared method given error_type

        Args:
            error_type (string): error type

        Return: 
            clean_method/dataset/model/scenario/compare_method:result

        """
        ## each error has two types of files
        # file type 1
        file1 = "delete" if error_type == "missing_values" else "dirty"
        file2 = list(set([k[3] for k in self.result.keys() if k[2] == error_type and k[3] != file1]))
        comparisons = {}
        metrics = {}

        for f2 in file2:
            file_types = [file1, f2]
            four_metrics = self.get_four_metrics(error_type, file_types)
            comparison = self.compare_four_metrics(error_type, four_metrics, file_types)
            metrics[f2] = four_metrics
            comparisons[f2] = comparison
        return comparisons, metrics

    def save_four_metrics(self, save_dir):
        for error_type in config.error_types:
            save_path = os.path.join(save_dir, "{}_four_metrics.xlsx".format(error_type['name']))
            utils.dfs_to_xls(self.four_metrics[error_type['name']], save_path)
        flat_metrics = utils.flatten_dict(self.four_metrics)

"""Comparing method"""
def t_test(dirty, clean):
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
     
    result = {}
    result['two_tail'] = two_tailed_t_test(dirty, clean)
    result['one_tail_pos'] = one_tailed_t_test(dirty, clean, 'positive')
    result['one_tail_neg'] = one_tailed_t_test(dirty, clean, 'negative')
    return result

def mean_f1(dirty, clean):
    result = {"dirty_f1": np.mean(dirty), "clean_f1":np.mean(clean)}
    return result

def mean_acc(dirty, clean):
    result = {"dirty_acc": np.mean(dirty), "clean_acc":np.mean(clean)}
    return result

def diff_f1(dirty, clean):
    result = {"diff_f1": np.mean((clean - dirty) / dirty)}
    return result

def diff_acc(dirty, clean):
    result = {"diff_acc": np.mean((clean - dirty) / dirty)}
    return result

def direct_count(dirty, clean):
    result = {"pos_count": np.sum(dirty - clean < -1e-8), "neg_count": np.sum(dirty - clean > 1e-8), "same_count": np.sum(np.abs(dirty - clean) < 1e-8)}
    return result

"""Comparing metrics"""
def test_f1(dataset_name, error_type, test_file):
    metric = test_file + "_test_f1"
    return metric

def test_acc(dataset_name, error_type, test_file):
    metric = test_file + "_test_acc"
    return metric

def mixed_f1_acc(dataset_name, error_type, test_file):
    if error_type == 'mislabel':
        dataset_name = dataset_name.split('_')[0]
    dataset = utils.get_dataset(dataset_name)
    if ('class_imbalance' in dataset.keys() and dataset['class_imbalance']):
        metric = test_file + "_test_f1"
    else:
        metric = test_file + "_test_acc"
    return metric

"""Multiple hypothesis test """
def hypothesis_test(t_test_results, alpha=0.05, multiple_test_method='fdr_by'):
    # convert to pd.DataFrame
    t_test_results_df = utils.dict_to_df(t_test_results, [0, 1, 2, 3, 4], [5, 6])

    # run BY procedure
    rejects = {}
    correct_p_vals = {}
    test_types = ['two_tail', 'one_tail_pos','one_tail_neg']
    pvals = [t_test_results_df.loc[:, (test_type, 'p-value')].values for test_type in test_types]
    pvals = np.concatenate(pvals, axis=0)
    rej, cor_p, m0, alpha_stages = multipletests(pvals, method=multiple_test_method, alpha=alpha)
    # print(np.max(pvals[rej]), np.max(cor_p[rej]))
    rej = np.split(rej, 3)
    cor_p = np.split(cor_p, 3)
    for test_type, r, p in zip(test_types, rej, cor_p):
        rejects[test_type] = pd.DataFrame(r, index=t_test_results_df.index, columns=['reject'])
        correct_p_vals[test_type] = pd.DataFrame(p, index=t_test_results_df.index, columns=['p-value'])

    hypothesis_result = {}
    for e, d, c, m, s, _, _ in t_test_results.keys():
        hypothesis_result[(e, d, c, m, s, 'two_tail_pvalue')] = correct_p_vals['two_tail'].loc[(e, d, c, m, s),'p-value']
        hypothesis_result[(e, d, c, m, s, 'pos_pvalue')] = correct_p_vals['one_tail_pos'].loc[(e, d, c, m, s),'p-value']
        hypothesis_result[(e, d, c, m, s, 'neg_pvalue')] = correct_p_vals['one_tail_neg'].loc[(e, d, c, m, s),'p-value']
        pos = rejects['one_tail_pos'].loc[(e, d, c, m, s), 'reject']
        neg = rejects['one_tail_neg'].loc[(e, d, c, m, s), 'reject']
        sig = rejects['two_tail'].loc[(e, d, c, m, s), 'reject']

        if sig and pos:
            hypothesis_result[(e, d, c, m, s, 'flag')] = 'P'
        elif sig and neg:
            hypothesis_result[(e, d, c, m, s, 'flag')] = 'N'
        else:
            hypothesis_result[(e, d, c, m, s, 'flag')] = 'S'
    return hypothesis_result

"""Group and split the result """
def split_clean_method(result):
    new_result = {}
    for (error, dataset, clean_method, model, scenario, comp_key), value in result.items():
        if error == 'outliers':
            detect = clean_method.split('_')[1]
            repair = clean_method.replace('_{}'.format(detect), '')
        else:
            detect = 'detect'
            repair = clean_method
        new_result[(error, dataset, detect, repair, model, scenario, comp_key)] = value
    return new_result

def group_by_mean(result):
    # group by training seed and reduce by mean
    result = utils.group(result, 5)
    result = utils.reduce_by_mean(result)
    return result

def group_by_best_model(result):
    # select best model by max val acc
    result = utils.group(result, 5)
    result = utils.reduce_by_max_val(result)
    result = utils.group(result, 4, keepdim=True)
    result = utils.reduce_by_max_val(result, dim=4, dim_name="model")
    return result

def group_by_best_model_clean(result):
    # select best model by max val acc
    result = utils.group_reduce_by_best_clean(result)
    return result    

def elim_redundant_dim(relation, dims):
    new_rel = {}
    for k, v in relation.items():
        new_key = tuple([k[i] for i in range(len(k)) if i not in dims])
        new_rel[new_key] = v
    return new_rel

"""Populate relations"""
def populate_relation(result, name, alphas=[0.05], split_detect=True, multiple_test_method='fdr_by'):
    print("Populate relation", name)
    # create save folder
    save_dir = utils.makedirs([config.analysis_dir, name])
    relation_dir = utils.makedirs([save_dir, 'relations'])
    metric_dir = utils.makedirs([save_dir, 'four_metrics'])

    # get other attributes
    attr_mean_acc = Compare(result, mean_acc, test_acc).compare_result       # attr: dirty_acc, clean_acc
    attr_diff_acc = Compare(result, diff_acc, test_acc).compare_result       # attr: diff_acc
    attr_mean_f1 = Compare(result, mean_f1, test_f1).compare_result          # attr: dirty_f1, clean_f1
    attr_diff_f1 = Compare(result, diff_f1, test_f1).compare_result          # attr: diff_f1
    attr_count = Compare(result, direct_count, mixed_f1_acc).compare_result  # attr: pos count, neg count, same count

    # run t-test 
    t_test_comp = Compare(result, t_test, mixed_f1_acc)
    t_test_comp.save_four_metrics(metric_dir)

    # hypothesis test
    for alpha in alphas:
        # print(alpha)
        # get attribute flag by multiple hypothesis test
        attr_flag = hypothesis_test(t_test_comp.compare_result, alpha, multiple_test_method)

        # populate relation with all of attributes
        relation = {**attr_flag, **attr_mean_acc, **attr_mean_f1, **attr_diff_acc, **attr_diff_f1, **attr_count}

        # split detect
        if split_detect and name != "R3":
            relation = split_clean_method(relation)

        # eliminate redundant attribute for R2 and R3

        if name == "R2":
            redundant_dims = [4] if split_detect else [3]
            relation = elim_redundant_dim(relation, redundant_dims)
        if name == "R3":
            redundant_dims = [2, 3]
            relation = elim_redundant_dim(relation, redundant_dims)
        
        # convert dict to df
        n_key = len(list(relation.keys())[0])
        relation_df = utils.dict_to_df(relation, list(range(n_key-1)), [n_key-1])

        # save relation to csv and pkl
        relation_csv_dir = utils.makedirs([relation_dir, 'csv'])
        save_path = os.path.join(relation_csv_dir, '{}_{}.csv'.format(name, "{:.6f}".format(alpha).rstrip('0')))
        relation_df.to_csv(save_path)

        relation_pkl_dir = utils.makedirs([relation_dir, 'pkl'])
        save_path = os.path.join(relation_pkl_dir, '{}_{}.pkl'.format(name, "{:.6f}".format(alpha).rstrip('0')))
        utils.df_to_pickle(relation_df, save_path)

def populate(alphas, save_training=False):
    """Populate R1, R2 and R3"""
    result = utils.load_result(parse_key=True)

    if save_training:
        save_dir = os.path.join(config.analysis_dir, "training_result")
        utils.result_to_table(result, save_dir)

    # populate R1
    result_mean = group_by_mean(result)
    populate_relation(result_mean, "R1", alphas=alphas)

    # populate R2
    result_best_model = group_by_best_model(result)
    populate_relation(result_best_model, "R2", alphas=alphas)

    # # populate R3
    result_best_model_clean = group_by_best_model_clean(result_best_model)
    populate_relation(result_best_model_clean, "R3", alphas=alphas)