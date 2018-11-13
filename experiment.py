from train import train
import numpy as np
import utils
import json
import argparse
import logging
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--log', default=False, action='store_true')
parser.add_argument('--nosave', default=False, action='store_true')
args = parser.parse_args()

if args.log:
    logging.captureWarnings(True)
    logging.basicConfig(filename='logging_{}.log'.format(datetime.datetime.now()),level=logging.DEBUG)

## Set up
# dataset_names = ["Titanic", "Marketing", "Airbnb", "EGG", "USCensus", "Sensor", "Credit"]
dataset_names = ['Airbnb', 'Movie', 'KDD']
model_names = [ "linear_regression", "logistic_regression", "decision_tree_regression", 
                "decision_tree_classification", "adaboost_classification", 
                "adaboost_regression", "knn_regression", "knn_classification", "random_forest_classification",
                "random_forest_regression", "guassian_naive_bayes"]

# dataset_names = ["Titanic"]
# model_names = ["logistic_regression"]

def get_coarse_grid(model):
    np.random.seed(args.seed)
    low, high = model["params_range"]
    if model["params_type"] == "real":
        param_grid = {model['params']: 10 ** np.random.uniform(low, high, 20)}
    if model["params_type"] == "int":
        param_grid = {model['params']: np.random.randint(low, high, 20)}
    return param_grid

def get_fine_grid(model, best_param_coarse):
    np.random.seed(args.seed)
    if model["params_type"] == "real":
        base = np.log10(best_param_coarse)
        param_grid = {model['params']: np.linspace(10**(base-0.5), 10**(base+0.5), 20)}
    if model["params_type"] == "int":
        low = max(best_param_coarse - 10, 1)
        param_grid = {model['params']: np.arange(low, low + 20)}
    return param_grid

## Training
result = utils.load_result()
datasets = [utils.get_dataset(d_name) for d_name in dataset_names]
models = [utils.get_model(m_name) for m_name in model_names]
jobs = [(d, e, f, m) for d in datasets for e in d["error_types"] for f in utils.get_filenames(e) for m in models]

for dataset, error_type, file, model in jobs:
    # Ignore if model type inconsitent with ml task
    dataset_name = dataset['data_dir']
    model_name = model['name']

    if model["type"] != dataset["ml_task"]:
        continue
    
    # Ignore if already trained
    key = "/".join((dataset_name, error_type, file, model_name, str(args.seed)))
    if key in result.keys():
        continue

    # Log info
    print("Processing {}".format(key))
    logging.info("Processing {}".format(key))
    
    estimator = model["estimator"]
    special = (dataset_name == 'IMDB')
    normalize = (model_name == 'linear_svm')

    if "params" not in model.keys():
        result_dict = train(dataset_name, error_type, file, estimator, None, args.cpu, special)
    else:
        # Coarse round
        param_grid = get_coarse_grid(model)
        result_coarse = train(dataset_name, error_type, file, estimator, param_grid, args.cpu, special=special, normalize=normalize)
        val_acc_coarse = result_coarse['val_acc']
        
        # Fine round
        best_param_coarse = result_coarse['best_params'][model['params']]
        param_grid = get_fine_grid(model, best_param_coarse)
        result_fine = train(dataset_name, error_type, file, estimator, param_grid, args.cpu, special)
        val_acc_fine = result_fine['val_acc']

        if val_acc_fine > val_acc_coarse:
            result_dict = result_fine
        else:
            result_dict = result_coarse
            
        if model["params_type"] == "int":
            result_dict['best_params'][model['params']] *= 1.0
    
    # print("Best params {}. Best val acc: {}".format(result_dict["best_params"], result_dict["val_acc"]))
    print(result_dict)
    if not args.nosave:
        utils.save_result(key, result_dict)