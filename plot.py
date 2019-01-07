import json
import pandas as pd
import numpy as np
import utils
from matplotlib import pyplot as plt
from matplotlib import patches
from table import *
import sys
import os

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'violet', 'grey', 'y']

def save_fig(save_dir):
    directory = os.path.dirname(save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(save_dir, bbox_inches='tight')
    plt.clf()

def filter(result, error_type):
    filtered_result = {k:v for k, v in result.items() if k[1] == error_type}
    return filtered_result

def seperate_keys(dic):
    datasets = sorted(set([k[0] for k in dic.keys()]))
    error_types = sorted(set([k[1] for k in dic.keys()]))
    methods = sorted(set([k[2] for k in dic.keys()]))
    models = sorted(set([k[3] for k in dic.keys()]))
    accs = sorted(set([k[4] for k in dic.keys()]))
    return datasets, error_types, methods, models, accs

def is_metric_f1(dataset_name):
    dataset = utils.get_dataset(dataset_name)
    return ('class_imbalance' in dataset.keys() and dataset['class_imbalance'])

def get_metric(dataset_name, test_file):
    if is_metric_f1(dataset_name):
        metric = test_file + "_test_f1"
    else:
        metric = test_file + "_test_acc"
    return metric

def get_ylabel(dataset, error_type):
    if error_type == 'missing_values':
        ylabel = "F1 Change Rate" if is_metric_f1(dataset) else "Accuracy Change Rate"
    else:
        ylabel = "F1 Change Rate" if is_metric_f1(dataset) else "Accuracy Change Rate"
    return ylabel

def bar_plot(data, xtic_labels, bar_names, xlabel, ylabel):
    n_bars, n_tics = data.shape
    x = list(range(n_tics))
    total_width = 0.8
    width = total_width / n_bars
    middle_name = bar_names[n_bars // 2]

    for row, name, c in zip(data, bar_names, colors):
        if name == middle_name:
            plt.bar(x, row, width=width, label=name, tick_label=xtic_labels, color=c)
        else:
            plt.bar(x, row, width=width, label=name, color=c)

        for i in range(len(x)):
            x[i] = x[i] + width
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    maximum = np.max(np.abs(data))
    ylim = max(0.1, maximum)
    plt.ylim((-ylim,ylim))

    ax = plt.gca()
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

def plot_legend(names, colors, title):
    # Create a color palette
    handles = [patches.Patch(color=c, label=x) for x, c in zip(names, colors)]
    # Create legend
    plt.legend(handles=handles, ncol=len(names), title=title)
    # Get current axes object and turn off axis
    plt.gca().set_axis_off()

def reindex_df(df, index_order, columns_order):
    df = df.reindex(index_order, axis=0)
    df = df.reindex(columns_order, axis=1)
    return df

def get_dataset_slice(df, dataset):
    data = df.loc[[dataset]].copy()
    index = list(data.index.droplevel(0))
    columns = [c[0] for c in df.columns]
    data = pd.DataFrame(data.values, index=index, columns=columns)
    return data

def compute_difference(evaluation):
    difference = {}
    for k, v in evaluation.items():
        dataset, method, model = k
        if method == 'dirty':
            continue
        difference[(dataset, method, model)] = (v - evaluation[(dataset, 'dirty', model)]) / evaluation[(dataset, 'dirty', model)] 
    return difference

def plot_column(result, index_order, columns_order, xtic_labels, bar_names, column_name, legend_title=None):
    datasets, error_types, methods, models, metrics = seperate_keys(result)
    evaluation = {}

    for dataset in datasets:
        for method in methods:
            for model in models:
                metric = get_metric(dataset, column_name)
                evaluation[(dataset, method, model)] = float(result[(dataset, error_types[0], method, model, metric)])

    difference = compute_difference(evaluation)
    df = utils.dict_to_df(difference, [0, 1], [2])
    for dataset in datasets:
        data = get_dataset_slice(df, dataset)
        data = reindex_df(data, index_order, columns_order)
        save_dir = "./plot/{}/{}_test/{}_{}_test.png".format(error_types[0], column_name, dataset, column_name)
        
        ylabel = get_ylabel(dataset, error_types[0])
        bar_plot(data.values, xtic_labels, bar_names, "Classification Models", ylabel)
        save_fig(save_dir)

    if legend_title is not None:
        plot_legend(bar_names, colors, legend_title)
        legend_dir = "./plot/{}/{}_test/{}_test_legend.png".format(error_types[0], column_name, column_name)
        save_fig(legend_dir)

    save_dir = "./plot/{}/{}_test.xls".format(error_types[0], column_name)
    utils.dict_to_xls(difference, [0, 1], [2], None, save_dir)

def plot_row(result, index_order, columns_order, xtic_labels, bar_names, row_name, legend_title=None):
    datasets, error_types, methods, models, metrics = seperate_keys(result)
    evaluation = {}

    for dataset in datasets:
        for model in models:
            dirty_metric = get_metric(dataset, 'dirty')
            clean_metric = get_metric(dataset, 'clean')
            evaluation[(dataset, 'dirty', model)] = float(result[(dataset, error_types[0], row_name, model, dirty_metric)])
            evaluation[(dataset, 'clean', model)] = float(result[(dataset, error_types[0], row_name, model, clean_metric)])

    difference = compute_difference(evaluation)
    df = utils.dict_to_df(difference, [0, 1], [2])
    for dataset in datasets:
        data = get_dataset_slice(df, dataset)
        data = reindex_df(data, index_order, columns_order)
        save_dir = "./plot/{}/{}_model/{}_{}_model.png".format(error_types[0], row_name, dataset, row_name)
        
        ylabel = get_ylabel(dataset, error_types[0])
        bar_plot(data.values, xtic_labels, bar_names, "Classification Models", ylabel)
        save_fig(save_dir)

    if legend_title is not None:
        plot_legend(bar_names, colors, legend_title)
        legend_dir = "./plot/{}/{}_model/{}_model_legend.png".format(error_types[0], row_name, row_name)
        save_fig(legend_dir)

    save_dir = "./plot/{}/{}_model.xls".format(error_types[0], row_name)
    utils.dict_to_xls(difference, [0, 1], [2], None, save_dir)

def plot_multirow_dirty(result, index_order, columns_order, xtic_labels, bar_names, legend_title=None):
    datasets, error_types, methods, models, metrics = seperate_keys(result)
    test_files = utils.get_filenames(error_types[0])
    clean_methods = sorted([f for f in test_files if f!='dirty'])[::-1]
    evaluation = {}

    for dataset in datasets:
        for model in models:
            dirty_metric = get_metric(dataset, 'dirty')
            clean_metric = get_metric(dataset, 'clean')
            evaluation[(dataset, 'dirty', model)] = float(result[(dataset, error_types[0], 'dirty', model, dirty_metric)])
            values = [float(v) for v in result[(dataset, error_types[0], 'dirty', model, clean_metric)].split('/')]
            for m, v in zip(clean_methods, values):
                evaluation[(dataset, m, model)] = v

    difference = compute_difference(evaluation)
    df = utils.dict_to_df(difference, [0, 1], [2])
    for dataset in datasets:
        data = get_dataset_slice(df, dataset)
        data = reindex_df(data, index_order, columns_order)
        save_dir = "./plot/{}/dirty_model/{}_dirty_model.png".format(error_types[0], dataset)
        
        ylabel = get_ylabel(dataset, error_types[0])
        bar_plot(data.values, xtic_labels, bar_names, "Classification Models", ylabel)
        save_fig(save_dir)
    
    if legend_title is not None:
        plot_legend(bar_names, colors, legend_title)
        legend_dir = "./plot/{}/dirty_model/dirty_model_legend.png".format(error_types[0])
        save_fig(legend_dir)

    save_dir = "./plot/{}/dirty_model.xls".format(error_types[0])
    utils.dict_to_xls(difference, [0, 1], [2], None, save_dir)

def plot_multirow_clean(result, index_order, columns_order, xtic_labels, bar_names, legend_title=None):
    datasets, error_types, methods, models, metrics = seperate_keys(result)
    test_files = utils.get_filenames(error_types[0])
    clean_methods = sorted([f for f in test_files if f!='dirty'])[::-1]
    evaluation = {}
    difference = {}

    for dataset in datasets:
        for model in models:
            dirty_metric = get_metric(dataset, 'dirty')
            clean_metric = get_metric(dataset, 'clean')
            for method in clean_methods:
                if error_types[0] == 'missing_values':
                    evaluation[(dataset, method, model, 'Delete Test')] = float(result[(dataset, error_types[0], method, model, dirty_metric)])
                    evaluation[(dataset, method, model, 'Impute Test')] = float(result[(dataset, error_types[0], method, model, clean_metric)])
                else:
                    evaluation[(dataset, method, model, 'Dirty Test')] = float(result[(dataset, error_types[0], method, model, dirty_metric)])
                    evaluation[(dataset, method, model, 'Clean Test')] = float(result[(dataset, error_types[0], method, model, clean_metric)])

                clean_res = float(result[(dataset, error_types[0], method, model, clean_metric)])
                dirty_res = float(result[(dataset, error_types[0], method, model, dirty_metric)])
                difference[(dataset, method, model)] = (clean_res - dirty_res) / dirty_res 

    df = utils.dict_to_df(difference, [0, 1], [2])
    for dataset in datasets:
        data = get_dataset_slice(df, dataset)
        data = reindex_df(data, index_order, columns_order)
        save_dir = "./plot/{}/clean_model/{}_clean_model.png".format(error_types[0], dataset)
        ylabel = get_ylabel(dataset, error_types[0])
        bar_plot(data.values, xtic_labels, bar_names, "Classification Models", ylabel)
        save_fig(save_dir)

    if legend_title is not None:
        plot_legend(bar_names, colors, legend_title)
        legend_dir = "./plot/{}/clean_model/clean_model_legend.png".format(error_types[0])
        save_fig(legend_dir)

    save_dir = "./plot/{}/clean_model.xls".format(error_types[0])
    utils.dict_to_xls(difference, [0, 1], [2], None, save_dir)

def plot_multicolumn_clean(result, index_order, columns_order, xtic_labels, bar_names, legend_title=None):
    datasets, error_types, methods, models, metrics = seperate_keys(result)
    test_files = utils.get_filenames(error_types[0])
    clean_methods = sorted([f for f in test_files if f!='dirty'])[::-1]
    evaluation = {}
    difference = {}

    for dataset in datasets:
        for model in models:
            clean_metric = get_metric(dataset, 'clean')
            values = [float(v) for v in result[(dataset, error_types[0], 'dirty', model, clean_metric)].split('/')]

            for m, v in zip(clean_methods, values):
                if error_types[0] == 'missing_values':
                    evaluation[(dataset, m, model, 'Delete Model')] = v
                    evaluation[(dataset, m, model, 'Impute Model')] = float(result[(dataset, error_types[0], m, model, clean_metric)])
                else:
                    evaluation[(dataset, m, model, 'Dirty Model')] = v
                    evaluation[(dataset, m, model, 'Clean Model')] = float(result[(dataset, error_types[0], m, model, clean_metric)])
                
                clean_res = float(result[(dataset, error_types[0], m, model, clean_metric)])
                difference[(dataset, m, model)] = (clean_res - v) / v 

    df = utils.dict_to_df(difference, [0, 1], [2])
    for dataset in datasets:
        data = get_dataset_slice(df, dataset)
        data = reindex_df(data, index_order, columns_order)
        save_dir = "./plot/{}/clean_test/{}_clean_test.png".format(error_types[0], dataset)
        plt.figure()
        ylabel = get_ylabel(dataset, error_types[0])
        bar_plot(data.values, xtic_labels, bar_names, "Classification Models", ylabel)
        save_fig(save_dir)

    if legend_title is not None:
        plot_legend(bar_names, colors, legend_title)
        legend_dir = "./plot/{}/clean_test/clean_test_legend.png".format(error_types[0])
        save_fig(legend_dir)

    save_dir = "./plot/{}/clean_test.xls".format(error_types[0])
    utils.dict_to_xls(difference, [0, 1], [2], None, save_dir)

def plot_outliers(summary):
    res_outlier = filter(summary, 'outliers')
    columns_order = ["logistic_regression", "knn_classification", "decision_tree_classification", "random_forest_classification", "adaboost_classification", "guassian_naive_bayes"]
    index_order = ["dirty", "clean_iso_forest_delete", "clean_iso_forest_impute_mean_dummy", "clean_iso_forest_impute_median_dummy", 
                            "clean_IQR_delete", "clean_IQR_impute_mean_dummy", "clean_IQR_impute_median_dummy",
                            "clean_SD_delete",  "clean_SD_impute_mean_dummy", "clean_SD_impute_median_dummy"]   
    xtic_labels = ["LR", "KNN", "DT", "RF", "AB", "NB"]
    bar_names = ["Dirty", "IF Delete", "IF Mean", "IF Median", "SD Delete", "SD Mean", "SD Median", "IQR Delete", "IQR Mean", "IQR Median"]
    legend_title = 'Cleaning Methods'

    plot_column(res_outlier, index_order[1:], columns_order, xtic_labels, bar_names[1:], 'dirty', legend_title=legend_title)
    plot_multicolumn_clean(res_outlier, index_order[1:], columns_order, xtic_labels, bar_names[1:], legend_title=legend_title)

    bar_names = ["Dirty", "IF Delete", "IF Mean", "IF Median", "SD Delete", "SD Mean", "SD Median", "IQR Delete", "IQR Mean", "IQR Median"]
    plot_multirow_dirty(res_outlier, index_order[1:], columns_order, xtic_labels, bar_names[1:], legend_title=legend_title)
    plot_multirow_clean(res_outlier, index_order[1:], columns_order, xtic_labels, bar_names[1:], legend_title=legend_title)

def plot_dup_incon(summary, error_type):
    res_duplicates = filter(summary, error_type)
    columns_order = ["logistic_regression", "knn_classification", "decision_tree_classification", "random_forest_classification", "adaboost_classification", "guassian_naive_bayes"]
    index_order = ["dirty", "clean"]   
    xtic_labels = ["LR", "KNN", "DT", "RF", "AB", "NB"]
    bar_names = ["Dirty Model", "Clean Model"]
    plot_column(res_duplicates, index_order[1:], columns_order, xtic_labels, bar_names[1:], 'dirty')
    plot_column(res_duplicates, index_order[1:], columns_order, xtic_labels, bar_names[1:], 'clean')

    bar_names = ["Dirty Test", "Clean Test"]
    plot_row(res_duplicates, index_order[1:], columns_order, xtic_labels, bar_names[1:], 'clean')
    plot_row(res_duplicates, index_order[1:], columns_order, xtic_labels, bar_names[1:], 'dirty')

def plot_mv(summary):
    res_mv = filter(summary, 'missing_values')
    columns_order = ["logistic_regression", "knn_classification", "decision_tree_classification", "random_forest_classification", "adaboost_classification", "guassian_naive_bayes"]
    index_order = ["dirty",
                    "clean_impute_mean_mode", 
                    "clean_impute_mean_dummy", 
                    "clean_impute_median_mode", 
                    "clean_impute_median_dummy", 
                    "clean_impute_mode_mode", 
                    "clean_impute_mode_dummy"]   
    xtic_labels = ["LR", "KNN", "DT", "RF", "AB", "NB"]
    bar_names = ["Delete", "Mean Mode", "Mean Dummy", "Median Mode", "Median Dummy", "Mode Dummy"]
    legend_title = 'Imputation Methods'

    plot_column(res_mv, index_order[1:], columns_order, xtic_labels, bar_names[1:], 'dirty', legend_title=legend_title)
    plot_multicolumn_clean(res_mv, index_order[1:], columns_order, xtic_labels, bar_names[1:], legend_title=legend_title)

    bar_names = ["Delete", "Mean Mode", "Mean Dummy", "Median Mode", "Median Dummy", "Mode Dummy"]
    plot_multirow_dirty(res_mv, index_order[1:], columns_order, xtic_labels, bar_names[1:], legend_title=legend_title)
    plot_multirow_clean(res_mv, index_order[1:], columns_order, xtic_labels, bar_names[1:], legend_title=legend_title)

raw_result = utils.load_result()
summary = summarize(raw_result, form_mean)
plot_outliers(summary)
plot_mv(summary)
plot_dup_incon(summary, 'duplicates')
plot_dup_incon(summary, 'inconsistency')