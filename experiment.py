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

## Set up
dataset_names = ["Titanic", "Marketing", "EGG", "USCensus", "Credit", "KDD", "Movie"]
model_names = [ "linear_regression", "logistic_regression", "decision_tree_regression", 
                "decision_tree_classification", "adaboost_classification", 
                "adaboost_regression", "knn_regression", "knn_classification", "random_forest_classification",
                "random_forest_regression", "guassian_naive_bayes"]

# dataset_names = ["Titanic"]
# model_names = ["logistic_regression"]

if args.log:
    logging.captureWarnings(True)
    logging.basicConfig(filename='logging_{}.log'.format(datetime.datetime.now()),level=logging.DEBUG)

def get_coarse_grid(model, random_state):
    np.random.seed(random_state)
    low, high = model["params_range"]
    if model["params_type"] == "real":
        param_grid = {model['params']: 10 ** np.random.uniform(low, high, 20)}
    if model["params_type"] == "int":
        param_grid = {model['params']: np.random.randint(low, high, 20)}
    return param_grid

def get_fine_grid(model, best_param_coarse, random_state):
    np.random.seed(random_state)
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
np.random.seed(args.seed)
seeds = np.random.randint(1000, size=3)
jobs = [(d, e, f, m, s) for d in datasets for e in d["error_types"] for f in utils.get_filenames(e) for m in models for s in seeds]

for dataset, error_type, file, model, seed in jobs:
    # Ignore if model type inconsitent with ml task
    dataset_name = dataset['data_dir']
    model_name = model['name']

    if model["type"] != dataset["ml_task"]:
        continue
    
    # Ignore if already trained
    key = "/".join((dataset_name, error_type, file, model_name, str(seed)))
    if key in result.keys():
        continue

    # Log info
    print("Processing {}".format(key))
    logging.info("Processing {}".format(key))
    
    estimator = model["estimator"]
    special = (dataset_name == 'IMDB')
    normalize = (model_name == 'linear_svm')

    np.random.seed(seed)
    coarse_param_seed, fine_param_seed, coarse_train_seed, fine_train_seed = np.random.randint(1000, size=4)

    if "params" not in model.keys():
        result_dict = train(dataset_name, error_type, file, estimator, None, n_jobs=args.cpu, special=special, normalize=normalize, random_state=coarse_param_seed)
    else:
        # Coarse round
        param_grid = get_coarse_grid(model, coarse_param_seed)
        result_coarse = train(dataset_name, error_type, file, estimator, param_grid, n_jobs=args.cpu, special=special, normalize=normalize, random_state=coarse_train_seed)
        val_acc_coarse = result_coarse['val_acc']
        
        # Fine round
        best_param_coarse = result_coarse['best_params'][model['params']]
        param_grid = get_fine_grid(model, best_param_coarse, fine_param_seed)
        result_fine = train(dataset_name, error_type, file, estimator, param_grid, n_jobs=args.cpu, special=special, normalize=normalize, random_state=fine_train_seed)
        val_acc_fine = result_fine['val_acc']

        if val_acc_fine > val_acc_coarse:
            result_dict = result_fine
        else:
            result_dict = result_coarse
            
        if model["params_type"] == "int":
            result_dict['best_params'][model['params']] *= 1.0

        result_dict['seeds'] = "{}/{}/{}/{}".format(coarse_param_seed, fine_param_seed, coarse_train_seed, fine_train_seed)
    
    print(result_dict)
    if not args.nosave:
        utils.save_result(key, result_dict)