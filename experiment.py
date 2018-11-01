from train import train
import numpy as np
import utils
from scipy.stats import expon
import json
import config

"""
Dataset: KDD, Citation, Marketing, Airbnb, DfD, Titanic, EGG, USCensus, Restaurant, Credit, Sensor, Movie, Food
error_type: missing_values, outliers, duplicates, inconsistency, mislabel
model: linear_regression, logistic_regression, knn, linear_svm, adaboost, neural_network, naive_bayes, robust
file:
    missing_values: 7
        clean_delete
        clean_impute_mean_mode, clean_impute_median_mode, clean_impute_mode_mode, 
        clean_impute_mean_dummy, clean_impute_median_dummy, clean_impute_mode_dummy
    outliers: 10
        dirty
        clean_IQR_delete, clean_IQR_impute_mean_dummy, clean_IQR_impute_median_dummy
        clean_SD_delete, clean_SD_impute_mean_dummy, clean_SD_impute_median_dummy
        clean_iso_forest_delete, clean_iso_forest_impute_mean_dummy, clean_iso_forest_impute_median_dummy
    duplicates: 2
        dirty, clean
    inconsistency: 2
        dirty, clean
    mislabel: 2
        dirty, clean
"""
np.random.seed(1)

datasets = ["Marketing", "Airbnb", "Titanic", "EGG", "USCensus", "Credit", "KDD"]
error_types = ["missing_values", "outliers", "duplicates", "inconsistency", "mislabel"]
files_dict = {  "missing_values": ["dirty", "clean_impute_mean_mode", "clean_impute_mean_dummy",  "clean_impute_median_mode", "clean_impute_median_dummy", "clean_impute_mode_mode", "clean_impute_mode_dummy"],
                "outliers": ["dirty", "clean_SD_delete", "clean_iso_forest_delete", "clean_IQR_delete", "clean_SD_impute_mean_dummy", "clean_IQR_impute_mean_dummy", "clean_iso_forest_impute_mean_dummy", "clean_SD_impute_median_dummy", "clean_IQR_impute_median_dummy", "clean_iso_forest_impute_median_dummy"],
                "duplicates":["dirty", "clean"],
                "inconsistency":["dirty", "clean"],
                "mislabel":["dirty", "clean"]}
result = json.load(open(config.result_dir, 'r'))

for dataset in datasets:
    d = utils.get_dataset(dataset)
    for error_type in d["error_types"]:
        for file in files_dict[error_type]:
            if d['ml_task'] == 'regression':
                model_name = "linear_regression"
            else:
                model_name = "logistic_regression"

            key = "/".join((dataset, error_type, model_name, file))

            if key in result.keys():
                continue

            print("Processing {}".format(key))
            model = utils.get_model(model_name)
            model_fn = model["fn"]
            params_dist = {model['params']:expon(scale=100)}
            best_param, train_acc, val_acc, test_acc = train(dataset, error_type, file, model_fn, params_dist)
            print("Best param {}. Best val acc: {}".format(best_param, val_acc))
            
            res = (best_param, train_acc, val_acc, test_acc)
            utils.save_result(key, res)