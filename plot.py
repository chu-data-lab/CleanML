""" Plot results"""
import json
import pandas as pd
import numpy as np
import utils
from table import *
from matplotlib import pyplot as plt
from matplotlib import patches
import sys
import os

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'violet', 'grey', 'y']
model_order = ["logistic_regression", "knn_classification", "decision_tree_classification", "random_forest_classification", "adaboost_classification", "guassian_naive_bayes"]
xtic_labels = ["LR", "KNN", "DT", "RF", "AB", "NB"]
xlabel = "ML Models"

def save_fig(save_dir):
    """ Save figure"""
    directory = os.path.dirname(save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(save_dir, bbox_inches='tight')
    plt.clf()

def bar_plot(data, xtic_labels, bar_names, xlabel, ylabel):
    """ Bar plot
        
        Args:
            data (numpy matrix): bars x xtics
            xtic_labels: labels for each tic
            bar_names: names for each bar
            xlabel: label for x axis
            ylabel: label for y axis
    """
    n_bars, n_tics = data.shape
    x = list(range(n_tics))
    total_width = 0.8
    width = total_width / n_bars
    middle_name = bar_names[n_bars // 2] # the name of bar in the middle

    for row, name, c in zip(data, bar_names, colors):
        if name == middle_name: 
            # put tick label under the middle bar
            plt.bar(x, row, width=width, label=name, tick_label=xtic_labels, color=c)
        else:
            plt.bar(x, row, width=width, label=name, color=c)

        for i in range(len(x)):
            x[i] = x[i] + width

    # set the center of y axis to be 0
    maximum = np.max(np.abs(data))
    ylim = max(0.1, maximum)
    plt.ylim((-ylim,ylim))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # set the value to be percentage
    ax = plt.gca()
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

def plot_legend(names, colors, title):
    """ Plot the legend by hand"""
    # Create a color palette
    handles = [patches.Patch(color=c, label=x) for x, c in zip(names, colors)]
    # Create legend
    plt.legend(handles=handles, ncol=len(names), title=title)
    # Get current axes object and turn off axis
    plt.gca().set_axis_off()

def get_ylabel(dataset):
    """ Get label for y axis"""
    ylabel = "Relative change of F1" if is_metric_f1(dataset) else "Relative change of Accuracy"
    return ylabel

def plot_comparisons(comparisons, error, methods, bar_names, legend_title=None):
    """ Plot comparison results
        
        Args:
            comparisons (dict): {clean_method: comparision result}
            error (string): error type
            methods (list): list of names of methods
            bar_names (list): list of names of bars on plot
            legend_title (string): title of legends
    """
    datasets = list(set(list(comparisons.values())[0].index.get_level_values(0)))
    comp_cases = list(set(list(comparisons.values())[0].columns.get_level_values(0)))
    for dataset in datasets:
        for comp in comp_cases:
            data = [comparisons[method].loc[dataset, comp].reindex(model_order, axis=0).values for method in methods]
            data = np.concatenate(data, axis=1).T
            bar_plot(data, xtic_labels, bar_names, xlabel, get_ylabel(dataset))
            save_dir = "./plot/{}/{}/{}_{}.png".format(error, comp, dataset, comp)
            save_fig(save_dir)

            if legend_title is not None:
                plot_legend(bar_names, colors, legend_title)
                legend_dir = "./plot/{}/{}/{}_legend.png".format(error, comp, comp)
                save_fig(legend_dir)

def plot_dup_incon(result, error):
    comparisons = compare_dup_incon(result, error)
    plot_comparisons(comparisons, error, ["clean"], ["clean"])

def plot_out(result):
    comparisons = compare_out(result)
    clean_methods = ["clean_SD_delete", "clean_SD_impute_mean_dummy", "clean_SD_impute_median_dummy",
                    "clean_IQR_delete", "clean_IQR_impute_mean_dummy", "clean_IQR_impute_median_dummy",
                    "clean_iso_forest_delete", "clean_iso_forest_impute_mean_dummy", "clean_iso_forest_impute_median_dummy"]
    bar_names = ["SD Delete", "SD Mean", "SD Median", "IQR Delete", "IQR Mean", "IQR Median", "IF Delete", "IF Mean", "IF Median"]
    plot_comparisons(comparisons, "outliers", clean_methods, bar_names, "Cleaning Methods")

def plot_mv(result):
    comparisons = compare_mv(result)
    impute_methods = ["clean_impute_mean_mode", "clean_impute_mean_dummy", 
                     "clean_impute_median_mode", "clean_impute_median_dummy", 
                     "clean_impute_mode_mode", "clean_impute_mode_dummy"]
    bar_names = ["Mean Mode", "Mean Dummy", "Median Mode", "Median Dummy", "Mode Mode", "Mode Dummy"]
    plot_comparisons(comparisons, "missing_values", impute_methods, bar_names, "Imputation Methods")

def plot_mislabel(result):
    comparisons = compare_mislabel(result)
    inject_methods = ["dirty_uniform", "dirty_major", "dirty_minor"]
    bar_names = ["Uniform", "Majority", "Minority"]
    plot_comparisons(comparisons, "mislabel", inject_methods, bar_names, "Injection Methods")

if __name__ == '__main__':
    result = utils.load_result()
    result = group_by_seed(result)
    result = reduce_by_mean(result)
    plot_dup_incon(result, "duplicates")
    plot_dup_incon(result, "inconsistency")
    plot_out(result)
    plot_mv(result)
    plot_mislabel(result)