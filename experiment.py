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

## setup
dataset_names = ["Citation"]
model_names = [ "logistic_regression", "decision_tree_classification", "adaboost_classification", 
                "knn_classification", "random_forest_classification", "guassian_naive_bayes"]

# dataset_names = ['KDD']
# model_names = ["guassian_naive_bayes"]

## save logging info
if args.log:
    logging.captureWarnings(True)
    logging.basicConfig(filename='logging_{}.log'.format(datetime.datetime.now()),level=logging.DEBUG)

def get_coarse_grid(model, random_state):
    """ Get hyper parameters (coarse grid) """
    np.random.seed(random_state)
    low, high = model["params_range"]
    if model["params_type"] == "real":
        param_grid = {model['params']: 10 ** np.random.uniform(low, high, 20)}
    if model["params_type"] == "int":
        param_grid = {model['params']: np.random.randint(low, high, 20)}
    return param_grid

def get_fine_grid(model, best_param_coarse, random_state):
    """ Get hyper parameters (fine grid, around the best parameter in coarse search) """
    np.random.seed(random_state)
    if model["params_type"] == "real":
        base = np.log10(best_param_coarse)
        param_grid = {model['params']: np.linspace(10**(base-0.5), 10**(base+0.5), 20)}
    if model["params_type"] == "int":
        low = max(best_param_coarse - 10, 1)
        param_grid = {model['params']: np.arange(low, low + 20)}
    return param_grid

def experiment(dataset, error_type, file, model, seed, result):
    """ Run experiment and save result
        
        Args:
            dataset (dict): dataset dict in config.py
            error_type (string): error type
            file (string): filename (dirty or clean) 
            model (dict): ml model dict in model.py
            seed (int): random seed
            result (dict): dict to save experimental results
                key (string): dataset_name/error_type/file/model_name/seed
                value (dict): metric_name: metric
    """

    # ignore if experiment has been runned before 
    key = "/".join((dataset['data_dir'], error_type, file, model['name'], str(seed)))
    if key in result.keys():
        print("Ignore experiment {} has been runned before.".format(key))
        return

    # print log info
    logging.info("Processing {}".format(key))
    print("Processing {}".format(key))

    # generate random seeds 
    np.random.seed(seed)
    coarse_param_seed, fine_param_seed, coarse_train_seed, fine_train_seed = np.random.randint(1000, size=4)
    
    estimator = model["estimator"]
    normalize = True # standarize data

    if "params" not in model.keys():
        # if no hyper parmeter, train directly
        result_dict = train(dataset, error_type, file, estimator, None, n_jobs=args.cpu, normalize=normalize, random_state=coarse_param_seed)
    else:
        # coarse search
        param_grid = get_coarse_grid(model, coarse_param_seed)
        result_coarse = train(dataset, error_type, file, estimator, param_grid, n_jobs=args.cpu, normalize=normalize, random_state=coarse_train_seed)
        val_acc_coarse = result_coarse['val_acc']
        
        # fine search
        best_param_coarse = result_coarse['best_params'][model['params']]
        param_grid = get_fine_grid(model, best_param_coarse, fine_param_seed)
        result_fine = train(dataset, error_type, file, estimator, param_grid, n_jobs=args.cpu, normalize=normalize, random_state=fine_train_seed)
        val_acc_fine = result_fine['val_acc']

        # save best result
        if val_acc_fine > val_acc_coarse:
            result_dict = result_fine
        else:
            result_dict = result_coarse
        
        # convert int to float to avoid json error
        if model["params_type"] == "int":
            result_dict['best_params'][model['params']] *= 1.0

        # save random seeds
        result_dict['seeds'] = "{}/{}/{}/{}".format(coarse_param_seed, fine_param_seed, coarse_train_seed, fine_train_seed)
    
    print(result_dict)
    # save result
    if not args.nosave:
        utils.save_result(key, result_dict)

if __name__ == '__main__':
    # load result dict
    result = utils.load_result()

    # get datasets and models
    datasets = [utils.get_dataset(d_name) for d_name in dataset_names]
    models = [utils.get_model(m_name) for m_name in model_names]

    # generate seed for 3 experiments
    np.random.seed(args.seed)
    seeds = np.random.randint(1000, size=3)

    # run experiments for each dataset, each error type, each model and each seed
    for d in datasets:
        for e in d["error_types"]:
            for f in utils.get_filenames(e):
                for m in models:
                    for s in seeds:
                        experiment(d, e, f, m, s, result)    