from train import train
import numpy as np
import utils
import json
import argparse
import logging
import datetime
import warnings

"""
Dataset: KDD, Citation, Marketing, Airbnb, DfD, Titanic, EGG, USCensus, Restaurant, Credit, Sensor, Movie, Food
error_type: missing_values, outliers, duplicates, inconsistency, mislabel
model: linear_regression, logistic_regression, knn, linear_svm, adaboost, neural_network, naive_bayes, robust
file:
    missing_values: 7
        dirty
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
parser = argparse.ArgumentParser()
parser.add_argument('--cpu', default=1, type=int)
parser.add_argument('--log', default=False, action='store_true')
parser.add_argument('--nosave', default=False, action='store_true')
args = parser.parse_args()
# warnings.simplefilter("ignore")
if args.log:
    logging.captureWarnings(True)
    logging.basicConfig(filename='logging_{}.log'.format(datetime.datetime.now()),level=logging.DEBUG)
np.random.seed(1)

# error_types = ["missing_values", "outliers", "duplicates", "inconsistency", "mislabel"]
files_dict = {  "missing_values": ["dirty", "clean_impute_mean_mode", "clean_impute_mean_dummy",  "clean_impute_median_mode", "clean_impute_median_dummy", "clean_impute_mode_mode", "clean_impute_mode_dummy"],
                "outliers": ["dirty", "clean_SD_delete", "clean_iso_forest_delete", "clean_IQR_delete", "clean_SD_impute_mean_dummy", "clean_IQR_impute_mean_dummy", "clean_iso_forest_impute_mean_dummy", "clean_SD_impute_median_dummy", "clean_IQR_impute_median_dummy", "clean_iso_forest_impute_median_dummy"],
                "duplicates":["dirty", "clean"],
                "inconsistency":["dirty", "clean"],
                "mislabel":["dirty", "clean"]}

# datasets = ["Marketing", "Airbnb", "Titanic", "EGG", "USCensus", "Sensor", "Credit", "KDD", "Movie"]
models = [  "linear_regression", "logistic_regression", "decision_tree_regression", 
            "decision_tree_classification", "linear_svm", "adaboost_classification", 
            "adaboost_regression", "knn_regression", "knn_classification", "random_forest_classification",
            "random_forest_regression", "guassian_naive_bayes"]

datasets = ["Titanic"]
models = ["logistic_regression"]

result = utils.load_result()

for dataset_name in datasets:
    dataset = utils.get_dataset(dataset_name)
    for error_type in dataset["error_types"]:
        files = files_dict[error_type]
        for file in files:
            for model_name in models:
                model = utils.get_model(model_name)
                if model["type"] != dataset["ml_task"]:
                    continue

                key = "/".join((dataset_name, error_type, file, model_name))
                if key in result.keys():
                    continue

                print("Processing {}".format(key))
                logging.info("Processing {}".format(key))
                
                estimator = model["estimator"]

                if "params" in model.keys():
                    # Coarse
                    if model["params_space"] == "log":
                        param_grid = {model['params']: 10 ** np.random.uniform(-5, 5, 20)}
                    if model["params_space"] == "int":
                        param_grid = {model['params']: np.random.randint(1, 100, 20)}
                    
                    special = (dataset_name == 'IMDB')
                    result_coarse = train(dataset_name, error_type, file, estimator, param_grid, args.cpu, special)
                    val_acc_coarse = result_coarse['val_acc']
                    
                    # Fine
                    best_param_coarse = result_coarse['best_params'][model['params']]
                    if model["params_space"] == "log":
                        base = np.log10(best_param_coarse)
                        param_grid = {model['params']: np.linspace(10**(base-0.5), 10**(base+0.5), 20)}
                    if model["params_space"] == "int":
                        low = max(best_param_coarse - 10, 1)
                        param_grid = {model['params']: np.linspace(low, low + 20, 20).astype(int)}

                    result_fine = train(dataset_name, error_type, file, estimator, param_grid, args.cpu, special)
                    val_acc_fine = result_fine['val_acc']

                    if val_acc_fine > val_acc_coarse:
                        result_dict = result_fine
                    else:
                        result_dict = result_coarse
                else:
                    result_dict = train(dataset_name, error_type, file, estimator, None, args.cpu, special)

                print("Best params {}. Best val acc: {}".format(result_dict["best_params"], result_dict["val_acc"]))
                if not args.nosave:
                    utils.save_result(key, result_dict)